import re
from pathlib import Path
from shutil import rmtree, which
from functools import partial
from contextlib import nullcontext

from beartype import beartype

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Dataset, random_split, Subset


from .model import ConditionalFlowMatcherWrapper
from .utils import STFTMag
from .data import get_dataloader
from .optimizer import get_optimizer

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

from tqdm import tqdm
import random
import torchaudio
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .utils import plot_tensor, save_plot_
from einops import rearrange
from scipy.signal import sosfiltfilt, cheby1, resample, resample_poly
from scipy.io.wavfile import write
import librosa 
import math
import os
import subprocess
import time
import matplotlib.pyplot as plt
from torchaudio.functional import resample as ta_resample

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

HIGHBAND_TRAIN_METRICS = (
    "mask_min",
    "mask_max",
    "mask_mean",
    "cutoff_hz",
    "cutoff_ratio",
    "residual_full_l1",
    "residual_low_l1",
    "residual_high_l1",
    "residual_hf_l1",
    "v_pred_hf_l1",
    "v_target_hf_l1",
    "pred_target_l1_ratio",
    "loss_lowband_debug",
    "loss_highband_debug",
    "loss_unmasked_debug",
    "loss_flow_masked",
    "loss_flow_hf_weighted",
    "loss_flow",
    "loss_hf_l1",
    "loss_hf_velocity_l1",
    "loss_edge_continuity",
    "loss_freq_pred",
    "loss_freq_grad",
    "freq_prediction_loss_weight",
    "freq_gradient_loss_weight",
    "use_freq_prediction_loss",
    "use_freq_gradient_loss",
    "loss_total_before_freq_aux",
    "loss_total_after_freq_aux",
    "loss_total",
    "loss_total_hfboost",
    "loss_melconv_bridge_l1",
    "loss_output_mel_adapter_l1",
    "segment_mask_valid_ratio",
    "melconv_bridge_mel_delta_l1",
    "melconv_bridge_hidden_delta_l1",
    "melconv_bridge_hidden_base_l1",
    "melconv_bridge_delta_base_ratio",
    "melconv_bridge_lowband_delta_l1",
    "melconv_bridge_scale",
    "melconv_bridge_ramp",
    "cbt_alpha",
    "cbt_event_gate_mean",
    "cbt_delta_l1",
    "cbt_injected_delta_l1",
    "cbt_transport_entropy",
    "cbt_active_high_4_6k",
    "cbt_active_high_6_8k",
    "cbt_active_high_8_12k",
    "cbt_active_high_12_16k",
    "cbt_active_high_16_24k",
    "output_adapter_delta_l1",
    "output_adapter_v_base_l1",
    "output_adapter_delta_base_ratio",
    "output_adapter_lowband_delta_l1",
    "output_adapter_near_delta_l1",
    "output_adapter_mid_delta_l1",
    "output_adapter_far_delta_l1",
    "output_adapter_band_gate_near",
    "output_adapter_band_gate_mid",
    "output_adapter_band_gate_far",
    "output_adapter_band_weight_mean",
    "output_adapter_band_weight_max",
    "output_adapter_ramp",
    "edge_bins",
    "hf_weight_min",
    "hf_weight_max",
    "adaln_gate_msa_mean",
    "adaln_gate_msa_abs_mean",
    "adaln_gate_ffn_mean",
    "adaln_gate_ffn_abs_mean",
)

HIGHBAND_VALID_METRICS = HIGHBAND_TRAIN_METRICS + (
    "lowband_delta",
    "hard_lowband_delta",
    "highband_delta",
    "LSD",
    "LSD_LF",
    "LSD_HF",
    "ViSQOL",
    "RTF",
    "NFEs",
)

def scalarize_metric(value):
    if isinstance(value, list):
        return float(np.mean(value)) if len(value) > 0 else 0.
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)

def append_metrics_txt(path, epoch, step, metrics):
    with open(path, "a", encoding="utf-8") as f:
        parts = []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, str):
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}={scalarize_metric(value):.8g}")
        ordered = " ".join(parts)
        f.write(f"epoch={epoch} step={step} {ordered}\n")

def filter_metric_keys(metrics, allowed_keys):
    if not allowed_keys:
        return metrics
    allowed_keys = set(allowed_keys)
    return {key: value for key, value in metrics.items() if key in allowed_keys}

def finite_metric_dict(metrics):
    safe = {}
    for key, value in metrics.items():
        try:
            scalar = scalarize_metric(value)
        except Exception:
            continue
        if math.isfinite(scalar):
            safe[key] = scalar
    return safe

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/flowhigh.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0
    return int(results[-1])



class TrainingHealthChecker:
    """Lightweight epoch-level health checker for high-band residual training.

    It consumes the merged epoch logs after validation and returns one of:
        continue / plateau / overfit / unstable
    """

    def __init__(
        self,
        patience=8,
        min_improve_ratio=0.01,
        overfit_patience=5,
        pred_ratio_range=(0.7, 1.2),
        hard_lowband_max=0.005,
        max_skip_rate=0.05,
    ):
        from collections import deque

        self.patience = int(patience)
        self.min_improve_ratio = float(min_improve_ratio)
        self.overfit_patience = int(overfit_patience)
        self.pred_ratio_min, self.pred_ratio_max = pred_ratio_range
        self.hard_lowband_max = float(hard_lowband_max)
        self.max_skip_rate = float(max_skip_rate)

        self.valid_losses = deque(maxlen=self.patience + 1)
        self.train_losses = deque(maxlen=self.patience + 1)
        self.skipped_steps = deque(maxlen=self.patience)
        self.best_valid = float("inf")
        self.bad_valid_epochs = 0

    @staticmethod
    def _finite(x):
        if x is None:
            return False
        try:
            value = scalarize_metric(x)
        except Exception:
            return False
        return math.isfinite(value)

    @staticmethod
    def _to_float(x, default=None):
        try:
            value = scalarize_metric(x)
        except Exception:
            return default
        return value if math.isfinite(value) else default

    def update(self, logs):
        train_loss = self._to_float(logs.get("train/loss_flow_masked"), None)
        if train_loss is None:
            train_loss = self._to_float(logs.get("training/cfm_loss"), None)
        valid_loss = self._to_float(logs.get("valid_avg/loss_flow_masked"), None)
        pred_ratio = self._to_float(logs.get("valid_avg/pred_target_l1_ratio"), None)
        hard_low = self._to_float(logs.get("valid_avg/hard_lowband_delta"), None)
        high_delta = self._to_float(logs.get("valid_avg/highband_delta"), None)
        low_delta = self._to_float(logs.get("valid_avg/lowband_delta"), None)
        skipped = self._to_float(logs.get("training/skipped_step", 0.0), 0.0)
        grad_norm = self._to_float(logs.get("training/grad_norm"), None)
        lsd_hf = self._to_float(logs.get("valid_avg/LSD_HF"), None)

        warnings = []
        status = "continue"

        if train_loss is not None:
            self.train_losses.append(train_loss)

        if valid_loss is None:
            return {
                "status": "unstable",
                "reason": "valid_avg/loss_flow_masked is missing or non-finite",
                "warnings": warnings,
                "best_valid": self.best_valid,
                "current_valid": valid_loss,
                "current_lsd_hf": lsd_hf,
            }

        self.valid_losses.append(valid_loss)
        if valid_loss < self.best_valid:
            self.best_valid = valid_loss
            self.bad_valid_epochs = 0
        else:
            self.bad_valid_epochs += 1

        self.skipped_steps.append(skipped if skipped is not None else 0.0)

        # Non-finite grad_norm is usually already skipped in train_step; flag it
        # here only if it appears in epoch logs.
        if grad_norm is not None and not math.isfinite(grad_norm):
            warnings.append("grad_norm is non-finite")
            status = "unstable"

        # Skipped-step rate over recent epochs.
        if len(self.skipped_steps) > 0:
            skip_rate = sum(self.skipped_steps) / max(1, len(self.skipped_steps))
            if skip_rate > self.max_skip_rate:
                warnings.append(f"skipped_step rate too high: {skip_rate:.3f}")
                status = "unstable"

        # Low-band pollution check.
        if hard_low is not None and hard_low > self.hard_lowband_max:
            warnings.append(f"hard_lowband_delta too high: {hard_low:.6f} > {self.hard_lowband_max}")
            status = "unstable"

        if high_delta is not None and low_delta is not None and high_delta <= low_delta * 2:
            warnings.append(
                f"highband_delta not sufficiently larger than lowband_delta: "
                f"high={high_delta:.6f}, low={low_delta:.6f}"
            )

        # Residual amplitude check.
        if pred_ratio is not None:
            if pred_ratio < self.pred_ratio_min:
                warnings.append(f"pred_target_l1_ratio too low: {pred_ratio:.3f}, possible residual collapse")
                status = "unstable"
            elif pred_ratio > self.pred_ratio_max:
                warnings.append(f"pred_target_l1_ratio too high: {pred_ratio:.3f}, possible noisy residual")
                status = "unstable"

        # Plateau check over recent validation losses.
        if len(self.valid_losses) >= self.patience + 1:
            old = self.valid_losses[0]
            new = self.valid_losses[-1]
            improve_ratio = (old - new) / max(abs(old), 1e-8)
            if improve_ratio < self.min_improve_ratio and status == "continue":
                status = "plateau"
                warnings.append(
                    f"valid loss improvement over last {self.patience} epochs is only "
                    f"{improve_ratio * 100:.2f}%"
                )

        # Overfit check: train decreases while valid increases over a window.
        if len(self.train_losses) >= self.overfit_patience + 1 and len(self.valid_losses) >= self.overfit_patience + 1:
            train_old = self.train_losses[-self.overfit_patience - 1]
            train_new = self.train_losses[-1]
            valid_old = self.valid_losses[-self.overfit_patience - 1]
            valid_new = self.valid_losses[-1]
            if train_new < train_old and valid_new > valid_old:
                status = "overfit"
                warnings.append(
                    f"train loss decreases but valid loss increases over last {self.overfit_patience} epochs"
                )

        return {
            "status": status,
            "reason": "; ".join(warnings) if warnings else "healthy",
            "warnings": warnings,
            "best_valid": self.best_valid,
            "current_valid": valid_loss,
            "current_lsd_hf": lsd_hf,
        }

class FLowHighTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        cfm_wrapper: ConditionalFlowMatcherWrapper,
        *,
        batch_size, dataset: Dataset, validset: Dataset,
        num_train_steps = None, num_warmup_steps = None, num_epochs = None,
        lr = 1e-4, initial_lr = 1e-5, grad_accum_every = 1, wd = 0., max_grad_norm = 0.5,
        valid_prepare = False, valid_frac = 0.05,
        random_split_seed = 53,
        log_every = 10, save_results_every = 100, save_model_every = 500, save_model_every_epochs = 5, results_folder = './results',
        force_clear_prev_results = None, split_batches = False, drop_last = False, 
        accelerate_kwargs: dict = dict(),
        original_sampling_rate = None, 
        tensorboard_logger = SummaryWriter,
        downsampling : str,
        sampling_rates : list,
        cfm_method = str,
        weighted_loss = False,
        model_name = str,
        validation_num_samples = 64,
        validation_time_steps = 1,
        fixed_validation_sample_rates = (8000, 12000, 16000),
        validate_on_epoch_end = True,
        val_step_interval = 0,
        cache_validation_batch = True,
        validation_save_audio = True,
        validation_save_audio_num = 4,
        save_best_checkpoint = True,
        best_metric_name = "valid_avg/loss_flow_masked",
        best_metric_mode = "min",
        find_unused_parameters = True,
        validation_batch_size = 1,
        tracker_project_name = "flowhigh",
        metrics_txt_path = None,
        validation_audio_dirname = "highband_validation",
        best_checkpoint_name = "best.pt",
        best_checkpoint_template = "best_epoch_{epoch:04d}_step_{step:06d}.pt",
        epoch_checkpoint_template = "FLowHigh.epoch{epoch}.step{step}.pt",
        final_checkpoint_template = "FLowHigh.final.{step}.pt",
        tensorboard_train_keys = None,
        tensorboard_valid_keys = None,
        txt_epoch_keys = None,
        scheduler_type = "cosine",
        bridge_lr = None,
        lite_transformer_lr = None,
        output_adapter_lr = None,
        input_sample_rates = None,
        sr_sampling_probs = None,
        compute_lsd_metrics = False,
        compute_visqol = False,
        visqol_bin = None,
    ):
        super().__init__()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = find_unused_parameters)
        self.accelerator = Accelerator(
            kwargs_handlers = [ddp_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )
        self.cfm_wrapper = cfm_wrapper
        audio_enc_dec = cfm_wrapper.flowhigh.audio_enc_dec
        self.register_buffer('steps', torch.Tensor([0]))
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.param_group_names = []
        bridge_lr_value = bridge_lr if bridge_lr is not None else lr
        lite_transformer_lr_value = lite_transformer_lr if lite_transformer_lr is not None else lr
        output_adapter_lr_value = output_adapter_lr if output_adapter_lr is not None else lr
        named_params = [(name, param) for name, param in cfm_wrapper.named_parameters() if param.requires_grad]
        bridge_params = [param for name, param in named_params if "melconv_bridge" in name or "cbt_bridge" in name]
        lite_transformer_params = [param for name, param in named_params if "block1_lite" in name or "second_transformer" in name]
        output_adapter_params = [param for name, param in named_params if "output_mel_adapter" in name]
        special_ids = {id(param) for param in bridge_params + lite_transformer_params + output_adapter_params}
        base_params = [param for _, param in named_params if id(param) not in special_ids]
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": lr, "name": "base"})
            self.param_group_names.append("base")
        if bridge_params:
            param_groups.append({"params": bridge_params, "lr": bridge_lr_value, "name": "bridge"})
            self.param_group_names.append("bridge")
        if lite_transformer_params:
            param_groups.append({"params": lite_transformer_params, "lr": lite_transformer_lr_value, "name": "lite_transformer"})
            self.param_group_names.append("lite_transformer")
        if output_adapter_params:
            param_groups.append({"params": output_adapter_params, "lr": output_adapter_lr_value, "name": "output_adapter"})
            self.param_group_names.append("output_adapter")
        self.optim = get_optimizer(param_groups, lr = lr, wd = wd, group_wd_params=False)
        self.param_groups_for_grad = {
            "base": base_params,
            "bridge": bridge_params,
            "lite_transformer": lite_transformer_params,
            "output_adapter": output_adapter_params,
        }
        self.param_group_base_lrs = [group["lr"] for group in param_groups]
        all_named_params = list(cfm_wrapper.named_parameters())

        def _count_params(params):
            return sum(param.numel() for param in params)

        block0_full_params = [param for name, param in all_named_params if "block0_full" in name]
        block1_lite_params = [param for name, param in all_named_params if "block1_lite" in name or "second_transformer" in name]
        melconv_bridge_all_params = [param for name, param in all_named_params if "melconv_bridge" in name]
        cbt_bridge_all_params = [param for name, param in all_named_params if "cbt_bridge" in name]
        output_adapter_all_params = [param for name, param in all_named_params if "output_mel_adapter" in name]
        ablation_special_ids = {id(param) for param in block0_full_params + block1_lite_params + melconv_bridge_all_params + cbt_bridge_all_params + output_adapter_all_params}
        base_all_params = [param for _, param in all_named_params if id(param) not in ablation_special_ids]

        print("num_params_total:", _count_params([param for _, param in all_named_params]))
        print("num_params_base:", _count_params(base_all_params))
        print("num_params_block0_full:", _count_params(block0_full_params))
        print("num_params_block1_lite:", _count_params(block1_lite_params))
        print("num_params_melconv_bridge:", _count_params(melconv_bridge_all_params))
        print("num_params_cbt_bridge:", _count_params(cbt_bridge_all_params))
        print("num_params_output_adapter:", _count_params(output_adapter_all_params))
        print("trainable_params_total:", _count_params([param for _, param in named_params]))
        print("base_lr:", lr)
        print("bridge_lr:", bridge_lr_value)
        print("lite_transformer_lr:", lite_transformer_lr_value)
        print("output_adapter_lr:", output_adapter_lr_value)
        print("num_base_params:", sum(param.numel() for param in base_params))
        print("num_bridge_params:", sum(param.numel() for param in bridge_params))
        print("num_lite_transformer_params:", sum(param.numel() for param in lite_transformer_params))
        print("num_output_adapter_params:", sum(param.numel() for param in output_adapter_params))
        self.lr = lr
        self.initial_lr = initial_lr

        # max grad norm
        self.max_grad_norm = max_grad_norm

        # create dataset
        self.ds = dataset

        # split for validation
        if valid_prepare:
            self.train_ds = self.ds
            self.valid_ds = validset
        else:
            if valid_frac > 0:
                train_size = int((1 - valid_frac) * len(self.ds))
                valid_size = len(self.ds) - train_size
                self.train_ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
                self.print(f'training with dataset of {len(self.train_ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')

        assert len(self.train_ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        self.steps_per_epoch = max(1, len(self.train_ds) // max(1, batch_size * grad_accum_every))

        assert exists(num_train_steps) or exists(num_epochs), 'either num_train_steps or num_epochs must be specified'

        if exists(num_epochs):
            self.num_train_steps = self.steps_per_epoch * num_epochs
        else:
            self.num_train_steps = num_train_steps
        self.num_epochs = math.ceil(self.num_train_steps / self.steps_per_epoch)
        print('num_train_stpes: ',num_train_steps)
        
        self.scheduler_type = scheduler_type
        if scheduler_type == "constant":
            self.scheduler = LambdaLR(self.optim, lr_lambda=lambda _: 1.0)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optim, T_max=self.num_train_steps)
        else:
            raise ValueError(f"unsupported scheduler_type: {scheduler_type}")
        self.num_warmup_steps = num_warmup_steps if exists(num_warmup_steps) else 0
        
        # dataloader
        self.dataloader = get_dataloader(self.train_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)
        self.valid_dataloader = get_dataloader(self.valid_ds, batch_size = validation_batch_size, shuffle = False, drop_last = drop_last)
        # fixed_valid_indices = [0, 11, 17, 31, 59, 61, 79, 83, 107, 119, 131, 151]  
        # fixed_valid_subset = Subset(self.valid_ds, fixed_valid_indices)
        # self.valid_dataloader = get_dataloader(fixed_valid_subset, batch_size = 1, shuffle = False, drop_last = drop_last)
        
        # prepare with accelerator 
        (self.cfm_wrapper, 
         self.optim, 
         self.scheduler, 
         self.dataloader
        ) = self.accelerator.prepare(
        self.cfm_wrapper, 
        self.optim, 
        self.scheduler,
        self.dataloader
        )

        # dataloader iterators
        self.dataloader_iter = cycle(self.dataloader)
        self.valid_dataloader_iter = cycle(self.valid_dataloader)

        # log & save
        self.log_every = log_every
        self.save_model_every = save_model_every
        self.save_model_every_epochs = save_model_every_epochs
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)
        print("results_folder",self.results_folder)
        self.log_txt_path = Path(metrics_txt_path) if exists(metrics_txt_path) else self.results_folder.parent / "log" / "metrics.txt"
        self.log_txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Ask if the existing checkpoint should be deleted
        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        # Create a directory for saving results
        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        # Hyperparameters for accelerator
        acc_hps = {
            "num_train_steps": self.num_train_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "learning_rate": self.lr,
            "initial_learning_rate": self.initial_lr,
            "wd": wd
        }
        self.accelerator.init_trackers(tracker_project_name, config=acc_hps)
        self.original_sampling_rate = original_sampling_rate
        self.tensorboard_logger = tensorboard_logger
        self.downsampling = downsampling
        self.sampling_rates = sampling_rates
        self.eval_stft = STFTMag(nfft=audio_enc_dec.n_fft,
                                 hop=audio_enc_dec.hop_length,
                                 window_len=audio_enc_dec.win_length)
        self.cfm_method = cfm_method
        self.weighted_loss = weighted_loss
        self.model_name = model_name
        self.validate_on_epoch_end = validate_on_epoch_end
        self.val_step_interval = val_step_interval
        self.validation_num_samples = len(self.valid_ds) if validation_num_samples is None or validation_num_samples <= 0 else validation_num_samples
        self.validation_time_steps = validation_time_steps
        self.fixed_validation_sample_rates = list(fixed_validation_sample_rates)
        self.cache_validation_batch = cache_validation_batch
        self.validation_save_audio = validation_save_audio
        self.validation_save_audio_num = validation_save_audio_num
        self.save_best_checkpoint = save_best_checkpoint
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode
        self.validation_audio_dirname = validation_audio_dirname
        self.best_checkpoint_name = best_checkpoint_name
        self.best_checkpoint_template = best_checkpoint_template
        self.epoch_checkpoint_template = epoch_checkpoint_template
        self.final_checkpoint_template = final_checkpoint_template
        self.tensorboard_train_keys = set(tensorboard_train_keys or [])
        self.tensorboard_valid_keys = set(tensorboard_valid_keys or [])
        self.txt_epoch_keys = list(txt_epoch_keys or [])
        self.input_sample_rates = list(input_sample_rates or [])
        self.sr_sampling_probs = None
        if exists(sr_sampling_probs):
            if len(self.input_sample_rates) != len(sr_sampling_probs):
                raise ValueError("sr_sampling_probs length must match input_sample_rates length")
            probs = np.asarray(sr_sampling_probs, dtype=np.float64)
            prob_sum = float(probs.sum())
            if prob_sum <= 0:
                raise ValueError("sr_sampling_probs must sum to a positive value")
            if not np.isclose(prob_sum, 1.0):
                self.print(f"[warning] sr_sampling_probs sum={prob_sum:.6f}; normalizing to 1.0")
                probs = probs / prob_sum
            self.sr_sampling_probs = probs.tolist()
        self.best_metric_value = None
        self.best_epoch = None
        self.best_step = None
        self.validation_cache = None
        self.epoch_metric_sums = {}
        self.epoch_metric_count = 0
        self.compute_lsd_metrics = compute_lsd_metrics
        self.compute_visqol = compute_visqol
        self.visqol_bin = visqol_bin
        self._visqol_warning_printed = False
        self.health_checker = TrainingHealthChecker(
            patience=8,
            min_improve_ratio=0.01,
            overfit_patience=5,
            pred_ratio_range=(0.7, 1.2),
            hard_lowband_max=0.005,
            max_skip_rate=0.05,
        )

        if self.cache_validation_batch:
            self.validation_cache = []
            for batch in self.valid_dataloader:
                self.validation_cache.append(batch)
                if len(self.validation_cache) >= self.validation_num_samples:
                    break

    def save_validset_txt(self):
        valid_data_paths = [self.ds[idx][1] for idx in self.valid_ds.indices]
        txt_path = self.results_folder / 'validation_dataset.txt'
        with open(txt_path, 'w') as f:
            for valid_data_path in valid_data_paths:
                f.write(f"{valid_data_path}\n")

        self.print(f"Validation dataset paths have been saved to {txt_path}")
    
    def save(self, path, full_state=True):
        if not self.is_main:
            return
        path = Path(path)
        tmp_path = path.with_name(path.name + ".tmp")
        model = self.accelerator.unwrap_model(self.cfm_wrapper)
        pkg = dict(
            model = model.state_dict(),
            step = int(self.steps.item()),
            epoch = int(self.steps.item()) // self.steps_per_epoch + 1,
        )
        if full_state:
            pkg.update(
                optim = self.optim.state_dict(),
                scheduler = self.scheduler.state_dict(),
                scaler = self.accelerator.scaler.state_dict() if exists(getattr(self.accelerator, "scaler", None)) else None,
            )
        try:
            torch.save(pkg, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            finally:
                raise

    def save_checkpoint(self, path, message_step=None, full_state=True):
        self.save(str(path), full_state=full_state)
        step_msg = "" if message_step is None else f"{message_step}: "
        self.print(f'{step_msg}saving model to {str(path)}')

    def load(self, path):
        cfm_wrapper = self.accelerator.unwrap_model(self.cfm_wrapper)
        pkg = cfm_wrapper.load(path, strict=not getattr(cfm_wrapper, "use_highband_residual_flow", False))

        self.optim.load_state_dict(pkg['optim'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        loaded_step = int(pkg.get('step', checkpoint_num_steps(path)))
        self.steps = torch.tensor([loaded_step + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)

    def unwrap_cfm_wrapper(self):
        return self.accelerator.unwrap_model(self.cfm_wrapper)

    def generate(self, *args, **kwargs):
        return self.cfm_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr

    def _log_metric_dict(self, prefix, metrics, step):
        scalar_metrics = {}
        for metric_name, value in metrics.items():
            scalar_value = scalarize_metric(value)
            full_key = f"{prefix}/{metric_name}"
            if self.tensorboard_valid_keys and full_key not in self.tensorboard_valid_keys:
                continue
            scalar_metrics[full_key] = scalar_value
            if self.is_main:
                self.tensorboard_logger.add_scalar(full_key, scalar_value, global_step=step)
        return scalar_metrics

    def _accumulate_epoch_metrics(self, metrics):
        for metric_name, value in metrics.items():
            self.epoch_metric_sums[metric_name] = self.epoch_metric_sums.get(metric_name, 0.) + scalarize_metric(value)
        self.epoch_metric_count += 1

    def _pop_epoch_metric_means(self):
        if self.epoch_metric_count == 0:
            return {}
        means = {
            metric_name: value / self.epoch_metric_count
            for metric_name, value in self.epoch_metric_sums.items()
        }
        self.epoch_metric_sums = {}
        self.epoch_metric_count = 0
        return means

    def _validation_batches(self):
        if exists(self.validation_cache):
            return self.validation_cache[:self.validation_num_samples]

        batches = []
        for _ in range(min(self.validation_num_samples, len(self.valid_ds))):
            batches.append(next(self.valid_dataloader_iter))
        return batches

    def _batch_to_hr_wave_and_length(self, batch):
        if len(batch) == 4:
            HR_wave, wav_length, _, _ = batch
        elif len(batch) == 2:
            HR_wave, wav_length = batch
        else:
            raise ValueError("validation expects dataset items: (HR_wave, wav_length) or (HR_wave, wav_length, up_cond, random_sr)")
        return HR_wave.to(self.device), wav_length.to(self.device)

    def _fixed_up_cond(self, hr_wave, fixed_input_sr, target_sr):
        down = ta_resample(hr_wave, target_sr, fixed_input_sr)
        up = ta_resample(down, fixed_input_sr, target_sr)
        if up.shape[-1] < hr_wave.shape[-1]:
            up = F.pad(up, (0, hr_wave.shape[-1] - up.shape[-1]))
        elif up.shape[-1] > hr_wave.shape[-1]:
            up = up[..., :hr_wave.shape[-1]]
        return up / up.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)

    def _sample_training_sr(self):
        if self.sr_sampling_probs is None:
            return None
        sampled = random.choices(self.input_sample_rates, weights=self.sr_sampling_probs, k=1)[0]
        return int(sampled)

    def _save_validation_audio(self, epoch, sample_index, fixed_input_sr, hr_wave, up_cond, ours_audio):
        sample_dir = self.results_folder / self.validation_audio_dirname / f"epoch_{epoch:04d}" / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_sr = self.unwrap_cfm_wrapper().flowhigh.audio_enc_dec.sampling_rate

        def save_wav(path, audio):
            audio_np = (audio.detach().cpu().squeeze().clamp(-1, 1).numpy() * 32767.0).astype(np.int16)
            write(str(path), target_sr, audio_np)

        sr_tag = f"{fixed_input_sr // 1000}k"
        save_wav(sample_dir / "gt.wav", hr_wave)
        save_wav(sample_dir / f"up_{sr_tag}.wav", up_cond)
        save_wav(sample_dir / f"ours_{sr_tag}.wav", ours_audio)

    def _match_audio_length(self, audio, target_length):
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[-1] < target_length:
            audio = F.pad(audio, (0, target_length - audio.shape[-1]))
        elif audio.shape[-1] > target_length:
            audio = audio[..., :target_length]
        return audio

    def _compute_lsd_metrics(self, reference_audio, predicted_audio, cutoff_hz):
        audio_enc_dec = self.unwrap_cfm_wrapper().flowhigh.audio_enc_dec
        target_sr = audio_enc_dec.sampling_rate
        n_fft = audio_enc_dec.n_fft
        hop_length = audio_enc_dec.hop_length
        win_length = audio_enc_dec.win_length

        reference_audio = self._match_audio_length(reference_audio, reference_audio.shape[-1]).float()
        predicted_audio = self._match_audio_length(predicted_audio, reference_audio.shape[-1]).float()
        window = torch.hann_window(win_length, device=reference_audio.device, dtype=reference_audio.dtype)

        ref_spec = torch.stft(
            reference_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs().clamp_min(1e-5)
        pred_spec = torch.stft(
            predicted_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs().clamp_min(1e-5)

        log_diff_sq = (torch.log(ref_spec) - torch.log(pred_spec)) ** 2
        lsd = torch.sqrt(log_diff_sq.mean()).detach()
        freqs = torch.linspace(0, target_sr / 2, ref_spec.shape[1], device=ref_spec.device, dtype=ref_spec.dtype)
        lf_mask = freqs <= cutoff_hz
        hf_mask = freqs > cutoff_hz

        lsd_lf = torch.sqrt(log_diff_sq[:, lf_mask, :].mean()).detach() if lf_mask.any() else torch.tensor(float("nan"), device=ref_spec.device)
        lsd_hf = torch.sqrt(log_diff_sq[:, hf_mask, :].mean()).detach() if hf_mask.any() else torch.tensor(float("nan"), device=ref_spec.device)
        return {
            "LSD": float(lsd.cpu()),
            "LSD_LF": float(lsd_lf.cpu()),
            "LSD_HF": float(lsd_hf.cpu()),
        }

    def _maybe_visqol_score(self, reference_audio, predicted_audio, target_sr):
        # ViSQOL is an optional external evaluator. When no binary is configured,
        # keep the metric visible as NaN instead of blocking the training run.
        if not self.compute_visqol or not self.visqol_bin:
            if self.compute_visqol and self.is_main and not self._visqol_warning_printed:
                self.print("ViSQOL enabled but no visqol_bin configured; logging ViSQOL=nan.")
                self._visqol_warning_printed = True
            return float("nan")

        visqol_cmd = str(self.visqol_bin)
        visqol_path = Path(visqol_cmd)
        if visqol_path.exists():
            visqol_cmd = str(visqol_path)
        elif which(visqol_cmd) is not None:
            visqol_cmd = which(visqol_cmd)
        else:
            if self.is_main and not self._visqol_warning_printed:
                self.print(f"ViSQOL binary not found: {self.visqol_bin}; logging ViSQOL=nan.")
                self._visqol_warning_printed = True
            return float("nan")

        tmp_dir = self.results_folder / "_visqol_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ref_path = tmp_dir / "ref.wav"
        deg_path = tmp_dir / "deg.wav"

        def save_tmp(path, audio):
            audio_np = (audio.detach().cpu().squeeze().clamp(-1, 1).numpy() * 32767.0).astype(np.int16)
            write(str(path), target_sr, audio_np)

        save_tmp(ref_path, reference_audio)
        save_tmp(deg_path, predicted_audio)
        try:
            proc = subprocess.run(
                [visqol_cmd, "--reference_file", str(ref_path), "--degraded_file", str(deg_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:
            return float("nan")

        match = re.search(r"(?:MOS-LQO|ViSQOL|score)[^0-9]*([0-9]+(?:\.[0-9]+)?)", proc.stdout + "\n" + proc.stderr, re.IGNORECASE)
        return float(match.group(1)) if match else float("nan")

    def _maybe_save_best_checkpoint(self, epoch, step, logs):
        if not self.save_best_checkpoint or self.best_metric_name not in logs:
            return None

        value = scalarize_metric(logs[self.best_metric_name])
        is_better = (
            self.best_metric_value is None
            or (self.best_metric_mode == "min" and value < self.best_metric_value)
            or (self.best_metric_mode == "max" and value > self.best_metric_value)
        )
        if not is_better:
            return None

        self.best_metric_value = value
        self.best_epoch = epoch
        self.best_step = step

        best_path = self.results_folder / self.best_checkpoint_name
        self.accelerator.wait_for_everyone()
        self.save_checkpoint(best_path, message_step=step, full_state=False)
        self.accelerator.wait_for_everyone()
        return {
            "best_metric_name": self.best_metric_name,
            "best_metric_value": value,
            "best_epoch": epoch,
            "best_step": step,
            "checkpoint_path": str(best_path),
        }

    @torch.inference_mode()
    def validate_highband_subset(self, step, epoch, fixed_input_sr, save_audio=False):
        unwrapped_model = self.accelerator.unwrap_model(self.cfm_wrapper)
        was_training = unwrapped_model.training
        unwrapped_model.eval()

        metric_sums = {}
        metric_count = 0
        target_sr = unwrapped_model.flowhigh.audio_enc_dec.sampling_rate
        generator = torch.Generator(device=self.device).manual_seed(1234 + int(fixed_input_sr))
        sr_tag = f"valid_{fixed_input_sr // 1000}k"

        for sample_index, batch in enumerate(self._validation_batches()):
            HR_wave, wav_length = self._batch_to_hr_wave_and_length(batch)
            up_cond = self._fixed_up_cond(HR_wave, fixed_input_sr, target_sr)

            win_length = unwrapped_model.flowhigh.audio_enc_dec.win_length
            hop_length = unwrapped_model.flowhigh.audio_enc_dec.hop_length
            mel_lengths = torch.ceil((wav_length - win_length) / hop_length + 1)

            with torch.no_grad():
                unwrapped_model.flowhigh.audio_enc_dec.eval()
                M_hr = unwrapped_model.flowhigh.audio_enc_dec.encode(HR_wave)
                M_up = unwrapped_model.flowhigh.audio_enc_dec.encode(up_cond)
                if M_hr.size(1) != M_up.size(1):
                    max_timelength = max(M_hr.size(1), M_up.size(1))
                    M_hr = F.pad(M_hr, (0, 0, max_timelength - M_hr.size(1), 0))
                    M_up = F.pad(M_up, (0, 0, max_timelength - M_up.size(1), 0))
                fixed_times = torch.rand((M_hr.shape[0],), dtype=M_hr.dtype, device=self.device, generator=generator)
                cutoff_hz = torch.full((M_hr.shape[0],), fixed_input_sr / 2, dtype=M_hr.dtype, device=self.device)
                mask_hf = unwrapped_model._build_hf_mask_for(M_hr, cutoff_hz).expand_as(M_hr)
                fixed_noise = torch.randn(M_hr.shape, dtype=M_hr.dtype, device=self.device, generator=generator)

                loss = unwrapped_model(
                M_hr,
                cond=M_up,
                cond_lengths=mel_lengths,
                cfm_method=self.cfm_method,
                random_sr=fixed_input_sr,
                weighted_loss=self.weighted_loss,
                validation_generator=generator,
                fixed_times=fixed_times,
                fixed_noise=fixed_noise,
                crop_mel_segments=False,
                global_step=step,
            )
            metrics = dict(getattr(unwrapped_model, "last_highband_debug", {}))
            inference_start = time.perf_counter()
            sample_debug = unwrapped_model.sample(
                cond=M_up,
                time_steps=self.validation_time_steps,
                decode_to_audio=False,
                random_sr=fixed_input_sr,
                cfm_method=self.cfm_method,
                return_intermediates=True,
                validation_generator=generator,
                initial_noise=fixed_noise,
            )
            ours_audio = unwrapped_model.flowhigh.audio_enc_dec.decode(sample_debug["M_final"])
            inference_elapsed = time.perf_counter() - inference_start
            ours_audio = self._match_audio_length(ours_audio, HR_wave.shape[-1])
            hr_for_metrics = self._match_audio_length(HR_wave, HR_wave.shape[-1])

            mask_hf = sample_debug["mask_hf"].expand_as(sample_debug["M_final"])
            mel_delta = sample_debug["M_final"] - sample_debug["M_up"]
            hard_low_mask = (mask_hf < 0.01).float()
            metrics.setdefault("loss_total", float(loss.detach().cpu()))
            metrics["loss_flow_masked"] = scalarize_metric(metrics.get("loss_flow_masked", loss))
            metrics["lowband_delta"] = float(torch.mean(torch.abs((1 - mask_hf) * mel_delta)).detach().cpu())
            metrics["hard_lowband_delta"] = float(torch.mean(torch.abs(hard_low_mask * mel_delta)).detach().cpu())
            metrics["highband_delta"] = float(torch.mean(torch.abs(mask_hf * mel_delta)).detach().cpu())
            duration_seconds = max(float(HR_wave.shape[-1]) / float(target_sr), 1e-6)
            metrics["RTF"] = float(inference_elapsed / duration_seconds)
            metrics["NFEs"] = float(self.validation_time_steps)
            if self.compute_lsd_metrics:
                metrics.update(self._compute_lsd_metrics(hr_for_metrics, ours_audio, fixed_input_sr / 2))
            if self.compute_visqol:
                metrics["ViSQOL"] = self._maybe_visqol_score(hr_for_metrics, ours_audio, target_sr)

            for metric_name in ("loss_flow_masked", *HIGHBAND_VALID_METRICS):
                if metric_name in metrics:
                    metric_sums[metric_name] = metric_sums.get(metric_name, 0.) + scalarize_metric(metrics[metric_name])
            metric_count += 1

            if save_audio and sample_index < self.validation_save_audio_num and fixed_input_sr == self.fixed_validation_sample_rates[0]:
                self._save_validation_audio(epoch, sample_index, fixed_input_sr, HR_wave, up_cond, ours_audio)

        if was_training:
            unwrapped_model.train()

        if metric_count == 0:
            return {}

        averaged = {key: value / metric_count for key, value in metric_sums.items()}
        accel_logs = self._log_metric_dict(sr_tag, averaged, step)
        self.accelerator.log(accel_logs, step=step)
        return averaged

    @torch.inference_mode()
    def validate_highband_fixed_cutoffs(self, epoch, step):
        all_logs = {}
        per_sr_logs = []

        for fixed_sr in self.fixed_validation_sample_rates:
            logs_sr = self.validate_highband_subset(
                step=step,
                epoch=epoch,
                fixed_input_sr=fixed_sr,
                save_audio=self.validation_save_audio
            )
            per_sr_logs.append(logs_sr)
            sr_tag = f"valid_{fixed_sr // 1000}k"
            all_logs.update({f"{sr_tag}/{key}": value for key, value in logs_sr.items()})

        avg_keys = (
            "loss_flow_masked",
            "loss_flow_hf_weighted",
            "loss_hf_l1",
            "loss_edge_continuity",
            "loss_freq_pred",
            "loss_freq_grad",
            "loss_total",
            "loss_total_hfboost",
            "loss_used_for_backward",
            "adaln_gate_msa_mean",
            "adaln_gate_msa_abs_mean",
            "adaln_gate_ffn_mean",
            "adaln_gate_ffn_abs_mean",
            "dynamic_hf_weight_max",
            "dynamic_hf_l1_weight",
            "trust_time_mean",
            "trust_time_low_mean",
            "trust_time_high_mean",
            "lowband_delta",
            "hard_lowband_delta",
            "highband_delta",
            "pred_target_l1_ratio",
            "melconv_bridge_delta_base_ratio",
            "melconv_bridge_lowband_delta_l1",
            "melconv_bridge_ramp",
            "cbt_alpha",
            "cbt_event_gate_mean",
            "cbt_delta_l1",
            "cbt_injected_delta_l1",
            "cbt_transport_entropy",
            "cbt_active_high_4_6k",
            "cbt_active_high_6_8k",
            "cbt_active_high_8_12k",
            "cbt_active_high_12_16k",
            "cbt_active_high_16_24k",
            "output_adapter_delta_base_ratio",
            "output_adapter_lowband_delta_l1",
            "output_adapter_near_delta_l1",
            "output_adapter_mid_delta_l1",
            "output_adapter_far_delta_l1",
            "output_adapter_ramp",
            "LSD",
            "LSD_LF",
            "LSD_HF",
            "ViSQOL",
            "RTF",
            "NFEs",
        )
        avg_logs = {}
        for key in avg_keys:
            vals = [logs[key] for logs in per_sr_logs if key in logs]
            if vals:
                avg_logs[key] = float(np.mean(vals))

        avg_tb_logs = self._log_metric_dict("valid_avg", avg_logs, step)
        self.accelerator.log(avg_tb_logs, step=step)
        if self.is_main:
            self.tensorboard_logger.flush()
        # 返回给 checkpoint / health checker 的日志不能只包含 TensorBoard 白名单，
        # 否则 Trust-Time 这类新实验会因为白名单未包含旧 key 而拿不到真实 validation loss。
        all_logs.update({f"valid_avg/{key}": value for key, value in avg_logs.items()})
        all_logs.update(avg_tb_logs)
        return all_logs

    def train_step(self):
        steps = int(self.steps.item())
        total_steps_per_epoch = max(1, len(self.train_ds) // (self.batch_size * self.grad_accum_every))
        current_epoch = steps // total_steps_per_epoch + 1
        self.cfm_wrapper.train()

        # Warmup is applied before the forward/backward pass.  The cosine
        # scheduler is stepped only after a successful optimizer.step().
        if steps < self.num_warmup_steps:
            warmup_frac = float(steps) / float(max(self.num_warmup_steps, 1))
            for param_group, base_lr in zip(self.optim.param_groups, self.param_group_base_lrs):
                param_group['lr'] = self.initial_lr + warmup_frac * (base_lr - self.initial_lr)

        logs = {}
        skipped_step = 0
        sr_count_logs = {}

        def _all_processes_finite(value):
            """Return True only if every distributed rank reports a finite value."""
            if not torch.is_tensor(value):
                value = torch.tensor(value, device=self.device)
            value = value.detach()
            finite_flag = torch.isfinite(value).all().to(torch.float32)
            if self.accelerator.num_processes > 1:
                finite_flag = self.accelerator.reduce(finite_flag, reduction="sum")
                return scalarize_metric(finite_flag) == float(self.accelerator.num_processes)
            return bool(scalarize_metric(finite_flag))

        # Gradient accumulation.
        for grad_accum_step in range(self.grad_accum_every):
            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.cfm_wrapper) if not is_last else nullcontext

            HR_wave, wav_length, up_cond, random_sr = next(self.dataloader_iter)
            unwrapped_model = self.unwrap_cfm_wrapper()
            win_length = unwrapped_model.flowhigh.audio_enc_dec.win_length
            hop_length = unwrapped_model.flowhigh.audio_enc_dec.hop_length
            mel_lengths = torch.ceil((wav_length - win_length) / hop_length + 1)
            sampled_sr = self._sample_training_sr()
            if exists(sampled_sr):
                # HF-Boost uses probability-sampled fixed SRs and rebuilds M_up
                # from HR_wave so condition and cutoff always match.
                random_sr = sampled_sr
                sr_count_logs[f"train/sr_{sampled_sr}_count"] = sr_count_logs.get(f"train/sr_{sampled_sr}_count", 0.0) + float(self.batch_size)
                up_cond = self._fixed_up_cond(HR_wave.to(self.device), sampled_sr, self.original_sampling_rate)
            else:
                up_cond = up_cond / up_cond.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)

            with self.accelerator.autocast(), context():
                loss = self.cfm_wrapper(
                    HR_wave,
                    cond=up_cond,
                    cond_lengths=mel_lengths,
                    cfm_method=self.cfm_method,
                    random_sr=random_sr,
                    weighted_loss=self.weighted_loss,
                    global_step=steps,
                )

                if not _all_processes_finite(loss):
                    skipped_step = 1
                    self.optim.zero_grad(set_to_none=True)
                    if self.is_main:
                        self.print(f"[skip] non-finite loss={scalarize_metric(loss)} at step={steps}")
                    logs['loss'] = scalarize_metric(loss.detach()) if torch.is_tensor(loss) else float(loss)
                    accel_logs = {
                        "train_loss": logs['loss'],
                        "training/cfm_loss": logs['loss'],
                        "training/lr": self.optim.param_groups[0]['lr'],
                        "training/skipped_step": 1.0,
                    }
                    self._accumulate_epoch_metrics(accel_logs)
                    self.steps += 1
                    return logs

                # IMPORTANT: `loss` is the scalar returned by cfm_wrapper.forward().
                # In HF-Boost mode it is exactly:
                # loss_flow_hf_weighted + loss_hf_l1_weight * loss_hf_l1.
                # Therefore this objective is used for back-propagation.
                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        grad_norm = None
        grad_norm_value = 0.0
        if exists(self.max_grad_norm):
            grad_norm = self.accelerator.clip_grad_norm_(self.cfm_wrapper.parameters(), self.max_grad_norm)
            grad_norm_value = scalarize_metric(grad_norm)
            if not _all_processes_finite(grad_norm):
                skipped_step = 1
                self.optim.zero_grad(set_to_none=True)
                if self.is_main:
                    self.print(f"[skip] non-finite grad_norm={grad_norm_value} at step={steps}")
        def _group_grad_norm(params):
            norms = []
            for param in params:
                if param.grad is not None:
                    norms.append(param.grad.detach().float().norm(2))
            if not norms:
                return 0.0
            return float(torch.stack(norms).norm(2).detach().cpu())
        grad_group_logs = {
            "train/grad_norm_total": grad_norm_value,
            "train/grad_norm_base": _group_grad_norm(self.param_groups_for_grad.get("base", [])),
            "train/grad_norm_bridge": _group_grad_norm(self.param_groups_for_grad.get("bridge", [])),
            "train/grad_norm_lite_transformer": _group_grad_norm(self.param_groups_for_grad.get("lite_transformer", [])),
            "train/grad_norm_output_adapter": _group_grad_norm(self.param_groups_for_grad.get("output_adapter", [])),
        }
        
        if not skipped_step:
            self.optim.step()
            if steps >= self.num_warmup_steps:
                self.scheduler.step()
            self.optim.zero_grad(set_to_none=True)

        current_lr = self.optim.param_groups[0]['lr']

        accel_logs = {
            "train_loss": logs.get('loss', 0.0),
            "training/cfm_loss": logs.get('loss', 0.0),
            "training/lr": current_lr,
            "training/skipped_step": float(skipped_step),
            "train/effective_batch_size": float(self.batch_size * self.grad_accum_every * self.accelerator.num_processes),
        }
        if exists(grad_norm):
            accel_logs["training/grad_norm"] = grad_norm_value
        accel_logs.update(grad_group_logs)
        highband_debug = getattr(self.accelerator.unwrap_model(self.cfm_wrapper), "last_highband_debug", {})
        if highband_debug:
            accel_logs["train/loss_flow_masked"] = scalarize_metric(highband_debug.get("loss_flow_masked", logs.get('loss', 0.0)))
            if "loss_used_for_backward" in highband_debug:
                accel_logs["train/loss_used_for_backward"] = scalarize_metric(highband_debug["loss_used_for_backward"])
            for metric_name in HIGHBAND_TRAIN_METRICS:
                if metric_name in highband_debug:
                    accel_logs[f"train/{metric_name}"] = scalarize_metric(highband_debug[metric_name])
        accel_logs.update(sr_count_logs)
        self._accumulate_epoch_metrics(accel_logs)

        # tensorboard / tracker / txt logging
        if not (steps % self.log_every):
            if self.is_main:
                train_tb_logs = filter_metric_keys(accel_logs, self.tensorboard_train_keys)
                for metric_name, metric_value in train_tb_logs.items():
                    self.tensorboard_logger.add_scalar(metric_name, metric_value, global_step=steps)
                self.tensorboard_logger.flush()
                self.print(
                    f"Epoch {current_epoch}, Step {steps}: "
                    f"loss: {logs.get('loss', 0.0):0.3f}, "
                    f"grad_norm: {grad_norm_value:0.3g}, "
                    f"skipped: {skipped_step}"
                )
            self.accelerator.log(filter_metric_keys(accel_logs, self.tensorboard_train_keys), step=steps)

        # sample results every so often
        self.accelerator.wait_for_everyone()

        # validation at fixed cutoff rates
        # All ranks must enter the same barrier here.  Only the main process runs
        # the expensive validation/generation, while worker ranks wait instead of
        # continuing into the next DDP backward pass.
        do_epoch_validation = self.validate_on_epoch_end and ((steps + 1) % self.steps_per_epoch == 0)
        do_step_validation = self.val_step_interval and steps > 0 and not (steps % self.val_step_interval)
        if do_epoch_validation or do_step_validation:
            self.accelerator.wait_for_everyone()
            valid_logs = {}
            if self.is_main:
                valid_logs = self.validate_highband_fixed_cutoffs(current_epoch, steps)
            epoch_train_logs = self._pop_epoch_metric_means()
            if self.is_main:
                if valid_logs:
                    valid_loss_for_print = valid_logs.get(
                        "valid_avg/loss_flow_masked",
                        valid_logs.get("valid_avg/loss_total", valid_logs.get("valid_avg/loss_flow_hf_weighted"))
                    )
                    valid_loss_text = "n/a" if valid_loss_for_print is None else f"{scalarize_metric(valid_loss_for_print):0.4f}"
                    lowband_text = "n/a" if "valid_avg/lowband_delta" not in valid_logs else f"{scalarize_metric(valid_logs['valid_avg/lowband_delta']):0.4f}"
                    highband_text = "n/a" if "valid_avg/highband_delta" not in valid_logs else f"{scalarize_metric(valid_logs['valid_avg/highband_delta']):0.4f}"
                    self.print(
                        f"Epoch {current_epoch}, Step {steps}: "
                        f"valid_loss: {valid_loss_text}, "
                        f"valid_avg/lowband_delta: {lowband_text}, "
                        f"valid_avg/highband_delta: {highband_text}"
                    )
                best_logs = self._maybe_save_best_checkpoint(current_epoch, steps, valid_logs)
                if best_logs:
                    valid_logs.update(best_logs)

                merged_epoch_logs = {**epoch_train_logs, **valid_logs}
                health = self.health_checker.update(merged_epoch_logs)
                status_id = {
                    "continue": 0,
                    "plateau": 1,
                    "overfit": 2,
                    "unstable": 3,
                }.get(health.get("status", "unknown"), -1)

                health_logs = {
                    "health/status_id": float(status_id),
                    "health/best_valid_loss": health.get("best_valid", float("nan")),
                    "health/current_valid_loss": health.get("current_valid", float("nan")),
                }
                if health.get("current_lsd_hf") is not None:
                    health_logs["health/current_lsd_hf"] = health["current_lsd_hf"]

                valid_logs.update(health_logs)
                self.print(
                    f"[HealthCheck] status={health.get('status')} | "
                    f"valid={health.get('current_valid')} | "
                    f"best={health.get('best_valid')} | "
                    f"reason={health.get('reason')}"
                )
                health_logs = finite_metric_dict(health_logs)
                for metric_name, metric_value in health_logs.items():
                    self.tensorboard_logger.add_scalar(metric_name, metric_value, global_step=steps)
                self.tensorboard_logger.flush()
                self.accelerator.log(health_logs, step=steps)

                append_metrics_txt(
                    self.log_txt_path,
                    current_epoch,
                    steps,
                    filter_metric_keys({**epoch_train_logs, **valid_logs}, self.txt_epoch_keys)
                )
                if current_epoch % self.save_model_every_epochs == 0:
                    epoch_model_path = self.results_folder / self.epoch_checkpoint_template.format(epoch=current_epoch, step=steps)
                    self.save_checkpoint(epoch_model_path, message_step=steps)
            self.accelerator.wait_for_everyone()

        self.steps += 1
        return logs

    def train(self, mid_ckpt=None, log_fn = noop):

        if mid_ckpt is not None:
            if os.path.exists(mid_ckpt):
                self.load(mid_ckpt)  
                print(f"Resuming training from checkpoint {mid_ckpt}")
            else:
                print(f"No checkpoint found at {mid_ckpt}, starting training from scratch")        
        else:
            print(f"starting training from scratch...")   

        start_step = int(self.steps.item())
        start_epoch = start_step // self.steps_per_epoch

        # One progress bar per epoch. This makes the progress inside the current
        # epoch visible instead of only updating after an epoch finishes.
        for epoch_idx in range(start_epoch, self.num_epochs):
            epoch_start_step = epoch_idx * self.steps_per_epoch
            epoch_end_step = min(self.num_train_steps, (epoch_idx + 1) * self.steps_per_epoch)

            current_step = int(self.steps.item())
            epoch_initial = max(0, current_step - epoch_start_step)
            epoch_total = max(1, epoch_end_step - epoch_start_step)

            epoch_loss_sum = 0.
            epoch_step_count = 0

            progress_bar = tqdm(
                total=epoch_total,
                initial=epoch_initial,
                desc=f"Epoch {epoch_idx + 1}/{self.num_epochs}",
                unit="step",
                disable=not self.is_local_main,
                dynamic_ncols=True,
                leave=True,
            )

            while int(self.steps.item()) < epoch_end_step:
                logs = self.train_step()
                log_fn(logs)

                loss_value = float(logs.get('loss', 0.))
                epoch_loss_sum += loss_value
                epoch_step_count += 1
                mean_epoch_loss = epoch_loss_sum / max(1, epoch_step_count)

                progress_bar.set_postfix(
                    step=int(self.steps.item()),
                    loss=f"{loss_value:.4f}",
                    avg=f"{mean_epoch_loss:.4f}",
                    lr=f"{self.optim.param_groups[0]['lr']:.2e}",
                )
                progress_bar.update(1)

            progress_bar.close()

        self.print('training complete')
        if self.is_main:
            final_path = self.results_folder / self.final_checkpoint_template.format(step=int(self.steps.item()))
            self.save_checkpoint(final_path)
            self.tensorboard_logger.flush()
            self.tensorboard_logger.close()
        self.accelerator.end_training()
