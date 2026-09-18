import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from data import AudioDataset
from config_highband_cbtbridge_full import config as hparams
from cfm_superresolution_highband import FLowHigh, MelVoco, ConditionalFlowMatcherWrapper
from trainer_highband import FLowHighTrainer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def split_full_dataset(dataset, train_ratio, seed):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)


def count_named_parameters(model, name_fragment):
    return sum(p.numel() for name, p in model.named_parameters() if name_fragment in name)


def print_active_high_gate_sanity(model):
    cbt_bridge = getattr(model, "cbt_bridge", None)
    if cbt_bridge is None:
        print("CBT active high gate sanity: cbt_bridge is disabled")
        return
    print("CBT active high gate sanity:")
    labels = ("4-6", "6-8", "8-12", "12-16", "16-24")
    for input_sr in (8000, 12000, 16000, 24000):
        gates = cbt_bridge.active_high_gates_for_cutoff(float(input_sr) / 2.0).squeeze(0).tolist()
        gate_text = ", ".join(f"{label}={int(round(value))}" for label, value in zip(labels, gates))
        print(f"  {input_sr // 1000}k: {gate_text}")


def print_cbt_init_sanity(model):
    cbt_bridge = getattr(model, "cbt_bridge", None)
    if cbt_bridge is None:
        print("CBTBridge init sanity: cbt_bridge is disabled")
        return
    print("CBTBridge init sanity:")
    print("  alpha:", float(cbt_bridge.alpha.detach().cpu()))
    print("  output_proj_abs_mean:", float(cbt_bridge.output_proj.weight.detach().abs().mean().cpu()))
    print("  output_proj_abs_max:", float(cbt_bridge.output_proj.weight.detach().abs().max().cpu()))


def resolve_experiment_path(path):
    """Resolve paths relative to this CBT package."""
    path = Path(path)
    if path.is_absolute():
        return path
    return EXPERIMENT_DIR / path


def configure_preferred_cuda_device():
    preferred_device = getattr(hparams.runtime, "preferred_cuda_device", None)
    if preferred_device is None:
        return None
    preferred_device = int(preferred_device)
    if not torch.cuda.is_available():
        return None
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        visible_list = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if str(preferred_device) not in visible_list:
            raise RuntimeError(
                f"Config prefers physical CUDA device {preferred_device}, but CUDA_VISIBLE_DEVICES={visible_devices!r}."
            )
        logical_device = visible_list.index(str(preferred_device))
    else:
        logical_device = preferred_device
    if logical_device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Configured CUDA device {preferred_device} maps to logical cuda:{logical_device}, "
            f"but torch only sees {torch.cuda.device_count()} CUDA device(s)."
        )
    torch.cuda.set_device(logical_device)
    return logical_device


def main():
    configured_cuda_device = configure_preferred_cuda_device()

    if hparams.runtime.require_cuda:
        assert torch.cuda.is_available(), "CPU training is not allowed."

    seed = hparams.runtime.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_dir = resolve_experiment_path(hparams.logging.log_dir)
    model_dir = resolve_experiment_path(hparams.checkpoint.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_txt_path = log_dir / hparams.logging.metrics_filename

    print("CBTBridge-Full startup")
    print("experiment_name: CBTBridge-Full / Lightweight Cross-Band Transport Bridge")
    print("data_path:", getattr(hparams.data, "data_path", None))
    print("train_path:", getattr(hparams.data, "train_path", None))
    print("valid_path:", getattr(hparams.data, "valid_path", None))
    print("test_path:", getattr(hparams.data, "test_path", None))
    print("test_split:", getattr(hparams.data, "test_split", None))
    print("checkpoint.save_dir:", hparams.checkpoint.save_dir)
    print("logging.log_dir:", hparams.logging.log_dir)
    print("logging.run_name:", hparams.logging.run_name)
    print("inference.model_path:", hparams.inference.model_path)
    print("train_from_scratch:", getattr(hparams.train, "train_from_scratch", None))
    print("resume_from:", hparams.train.resume_from)
    print("preferred_cuda_visible_devices:", getattr(hparams.runtime, "preferred_cuda_visible_devices", None))
    print("preferred_cuda_device:", getattr(hparams.runtime, "preferred_cuda_device", None))
    print("configured_torch_cuda_device:", configured_cuda_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("use_interleaved_melconv:", getattr(hparams.model, "use_interleaved_melconv", None))
    print("Bridge enabled:", getattr(hparams.model, "use_melconv_bridge", None))
    print("cbt_bridge_enabled:", getattr(hparams.model, "cbt_bridge_enabled", None))
    print("cbt_hidden_dim:", getattr(hparams.model, "cbt_hidden_dim", None))
    print("cbt_low_groups_hz:", getattr(hparams.model, "cbt_low_groups_hz", None))
    print("cbt_high_groups_hz:", getattr(hparams.model, "cbt_high_groups_hz", None))
    print("cbt_zero_init:", getattr(hparams.model, "cbt_zero_init", None))
    print("use_output_mel_adapter:", getattr(hparams.model, "use_output_mel_adapter", None))
    print("OutputAdapter enabled:", getattr(hparams.model, "use_output_mel_adapter", None))
    print("HeadAdapter enabled:", getattr(hparams.model, "use_output_mel_adapter", None))
    print("input_sample_rates:", hparams.train.input_sample_rates)
    print("sr_sampling_probs:", hparams.train.sr_sampling_probs)
    print("model dim:", getattr(hparams.model, "dim", None))
    print("model depth:", getattr(hparams.model, "n_layers", None))
    print("model heads:", getattr(hparams.model, "n_heads", None))
    print("model dim_head:", getattr(hparams.model, "dim_head", None))
    print("first_transformer_heads:", getattr(hparams.model, "first_transformer_heads", None))
    print("first_transformer_depth:", getattr(hparams.model, "first_transformer_depth", None))
    print("first_transformer_ff_mult:", getattr(hparams.model, "first_transformer_ff_mult", None))
    print("second_transformer_heads:", getattr(hparams.model, "second_transformer_heads", None))
    print("second_transformer_depth:", getattr(hparams.model, "second_transformer_depth", None))
    print("second_transformer_ff_mult:", getattr(hparams.model, "second_transformer_ff_mult", None))
    print("melconv_bridge_channels:", getattr(hparams.model, "melconv_bridge_channels", None))
    print("output_mel_adapter_channels:", getattr(hparams.model, "output_mel_adapter_channels", None))
    print("resolved_output_dir:", model_dir)
    print("learning_rate:", hparams.train.lr)
    print("bridge_lr:", getattr(hparams.train, "bridge_lr", hparams.train.lr))
    print("lite_transformer_lr:", getattr(hparams.train, "lite_transformer_lr", hparams.train.lr))
    print("output_adapter_lr:", getattr(hparams.train, "output_adapter_lr", hparams.train.lr))
    print("scheduler:", hparams.train.scheduler_type)
    print("warmup_steps:", hparams.train.n_warmup_steps)
    print("hf_weight_min/max:", hparams.train.hf_weight_min, hparams.train.hf_weight_max)
    print("hf_weight_mode:", getattr(hparams.train, "hf_weight_mode", "linear"))
    print("hf_weight_cap_hz_above_cutoff:", getattr(hparams.train, "hf_weight_cap_hz_above_cutoff", 4000.0))
    print("loss_hf_l1_weight:", hparams.train.loss_hf_l1_weight)
    print("edge_continuity_loss_weight:", getattr(hparams.train, "edge_continuity_loss_weight", 0.0))
    print("residual_noise_scale:", getattr(hparams.model, "residual_noise_scale", 1.0))
    print("use_freq_prediction_loss:", getattr(hparams.train, "use_freq_prediction_loss", False))
    print("freq_prediction_loss_weight:", getattr(hparams.train, "freq_prediction_loss_weight", 0.0))
    print("use_freq_gradient_loss:", getattr(hparams.train, "use_freq_gradient_loss", False))
    print("freq_gradient_loss_weight:", getattr(hparams.train, "freq_gradient_loss_weight", 0.0))
    print("use_adaln_zero:", getattr(hparams.model, "use_adaln_zero", False))
    print("input_sample_rates:", hparams.train.input_sample_rates)
    print("sr_sampling_probs:", hparams.train.sr_sampling_probs)
    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    print("batchsize:", hparams.train.batchsize)
    print("grad_accum_every:", hparams.train.grad_accum_every)
    print("num_processes:", num_processes)
    print("per_process_effective_batch:", hparams.train.batchsize * hparams.train.grad_accum_every)
    print("global_effective_batch:", hparams.train.batchsize * hparams.train.grad_accum_every * num_processes)
    print("num_epochs:", hparams.train.num_epochs)
    print("best_metric_name:", hparams.checkpoint.best_metric_name)
    print("Num of current cuda devices:", torch.cuda.device_count())

    logger = SummaryWriter(log_dir=str(log_dir))
    try:
        print("Audio extension:", hparams.data.audio_extension)
        print("Audio loader uses recursive search under:", hparams.data.data_path)
        full_dataset = AudioDataset(
            folder=hparams.data.data_path,
            audio_extension=hparams.data.audio_extension,
            downsampling=hparams.data.downsampling_method,
        )
        if len(full_dataset) == 0:
            raise RuntimeError(f"No audio files found under {hparams.data.data_path}")

        train_dataset, valid_dataset = split_full_dataset(
            full_dataset,
            train_ratio=hparams.data.train_split,
            seed=hparams.data.random_split_seed,
        )
        print("Total audio files found:", len(full_dataset))
        print("Train split:", len(train_dataset))
        print("Valid split from official100 train folder:", len(valid_dataset))

        sampling_rates = list(range(
            hparams.data.downsample_min,
            hparams.data.downsample_max + hparams.data.downsample_step,
            hparams.data.downsample_step,
        ))

        audio_enc_dec = MelVoco(
            n_mels=hparams.data.n_mel_channels,
            sampling_rate=hparams.data.samplingrate,
            f_max=hparams.data.mel_fmax,
            f_min=hparams.data.mel_fmin,
            n_fft=hparams.data.n_fft,
            win_length=hparams.data.win_length,
            hop_length=hparams.data.hop_length,
            vocoder=hparams.model.vocoder,
            vocoder_config=hparams.model.vocoderconfigpath,
            vocoder_path=hparams.model.vocoderpath,
        )

        model = FLowHigh(
            architecture=hparams.model.architecture,
            dim_in=hparams.data.n_mel_channels,
            audio_enc_dec=audio_enc_dec,
            dim=hparams.model.dim,
            depth=hparams.model.n_layers,
            dim_head=hparams.model.dim_head,
            heads=hparams.model.n_heads,
            input_channels=hparams.model.input_channels,
            condition_with_hf_mask=getattr(hparams.model, "condition_with_hf_mask", False),
            condition_with_cutoff_embedding=getattr(hparams.model, "condition_with_cutoff_embedding", False),
            use_adaln_zero=getattr(hparams.model, "use_adaln_zero", False),
            adaln_zero_init=getattr(hparams.model, "adaln_zero_init", True),
            adaln_zero_gate_log=getattr(hparams.model, "adaln_zero_gate_log", True),
            use_interleaved_melconv=getattr(hparams.model, "use_interleaved_melconv", False),
            use_melconv_bridge=getattr(hparams.model, "use_melconv_bridge", False),
            cbt_bridge_enabled=getattr(hparams.model, "cbt_bridge_enabled", False),
            cbt_hidden_dim=getattr(hparams.model, "cbt_hidden_dim", 32),
            cbt_low_groups_hz=getattr(hparams.model, "cbt_low_groups_hz", None),
            cbt_high_groups_hz=getattr(hparams.model, "cbt_high_groups_hz", None),
            cbt_use_event_gate=getattr(hparams.model, "cbt_use_event_gate", True),
            cbt_use_temporal_derivative=getattr(hparams.model, "cbt_use_temporal_derivative", True),
            cbt_use_depthwise_temporal_conv=getattr(hparams.model, "cbt_use_depthwise_temporal_conv", True),
            cbt_temporal_kernel=getattr(hparams.model, "cbt_temporal_kernel", 5),
            cbt_zero_init=getattr(hparams.model, "cbt_zero_init", True),
            cbt_init_scale=getattr(hparams.model, "cbt_init_scale", 0.0),
            cbt_dropout=getattr(hparams.model, "cbt_dropout", 0.0),
            use_output_mel_adapter=getattr(hparams.model, "use_output_mel_adapter", False),
            first_transformer_depth=getattr(hparams.model, "first_transformer_depth", 1),
            first_transformer_heads=getattr(hparams.model, "first_transformer_heads", hparams.model.n_heads),
            first_transformer_dim_head=getattr(hparams.model, "first_transformer_dim_head", hparams.model.dim_head),
            first_transformer_ff_mult=getattr(hparams.model, "first_transformer_ff_mult", 4),
            second_transformer_depth=getattr(hparams.model, "second_transformer_depth", 1),
            second_transformer_heads=getattr(hparams.model, "second_transformer_heads", 8),
            second_transformer_dim_head=getattr(hparams.model, "second_transformer_dim_head", hparams.model.dim_head),
            second_transformer_ff_mult=getattr(hparams.model, "second_transformer_ff_mult", 2),
            melconv_bridge_channels=getattr(hparams.model, "melconv_bridge_channels", 64),
            melconv_bridge_blocks=getattr(hparams.model, "melconv_bridge_blocks", 3),
            melconv_bridge_kernel_time=getattr(hparams.model, "melconv_bridge_kernel_time", 3),
            melconv_bridge_kernel_freq=getattr(hparams.model, "melconv_bridge_kernel_freq", 9),
            melconv_bridge_zero_init=getattr(hparams.model, "melconv_bridge_zero_init", False),
            melconv_bridge_final_init_std=getattr(hparams.model, "melconv_bridge_final_init_std", 1e-4),
            melconv_bridge_scale_init=getattr(hparams.model, "melconv_bridge_scale_init", 1.0),
            melconv_bridge_scale_ramp_steps=getattr(hparams.model, "melconv_bridge_scale_ramp_steps", 5000),
            melconv_bridge_use_hidden_mel=getattr(hparams.model, "melconv_bridge_use_hidden_mel", True),
            melconv_bridge_use_zt=getattr(hparams.model, "melconv_bridge_use_zt", True),
            melconv_bridge_use_cond=getattr(hparams.model, "melconv_bridge_use_cond", True),
            melconv_bridge_use_mask=getattr(hparams.model, "melconv_bridge_use_mask", True),
            melconv_bridge_use_freq_pos=getattr(hparams.model, "melconv_bridge_use_freq_pos", True),
            melconv_bridge_use_cutoff_dist=getattr(hparams.model, "melconv_bridge_use_cutoff_dist", True),
            output_mel_adapter_channels=getattr(hparams.model, "output_mel_adapter_channels", 64),
            output_mel_adapter_blocks=getattr(hparams.model, "output_mel_adapter_blocks", 2),
            output_mel_adapter_kernel_time=getattr(hparams.model, "output_mel_adapter_kernel_time", 3),
            output_mel_adapter_kernel_freq=getattr(hparams.model, "output_mel_adapter_kernel_freq", 9),
            output_mel_adapter_zero_init=getattr(hparams.model, "output_mel_adapter_zero_init", False),
            output_mel_adapter_final_init_std=getattr(hparams.model, "output_mel_adapter_final_init_std", 1e-4),
            output_mel_adapter_scale_ramp_steps=getattr(hparams.model, "output_mel_adapter_scale_ramp_steps", 5000),
            output_mel_adapter_use_v_base=getattr(hparams.model, "output_mel_adapter_use_v_base", True),
            output_mel_adapter_use_zt=getattr(hparams.model, "output_mel_adapter_use_zt", True),
            output_mel_adapter_use_cond=getattr(hparams.model, "output_mel_adapter_use_cond", True),
            output_mel_adapter_use_mask=getattr(hparams.model, "output_mel_adapter_use_mask", True),
            output_mel_adapter_use_freq_pos=getattr(hparams.model, "output_mel_adapter_use_freq_pos", True),
            output_mel_adapter_use_cutoff_dist=getattr(hparams.model, "output_mel_adapter_use_cutoff_dist", True),
            output_mel_adapter_use_band_gates=getattr(hparams.model, "output_mel_adapter_use_band_gates", True),
            output_mel_adapter_near_hz=getattr(hparams.model, "output_mel_adapter_near_hz", 4000.0),
            output_mel_adapter_mid_hz=getattr(hparams.model, "output_mel_adapter_mid_hz", 10000.0),
            output_mel_adapter_band_gate_init=getattr(hparams.model, "output_mel_adapter_band_gate_init", 1.0),
        )

        cfm_wrapper = ConditionalFlowMatcherWrapper(
            flowhigh=model,
            cfm_method=hparams.model.cfm_path,
            sigma=hparams.model.sigma,
            use_highband_residual_flow=getattr(hparams.model, "use_highband_residual_flow", False),
            highband_mask_softness_hz=getattr(hparams.model, "highband_mask_softness_hz", 500.0),
            condition_with_hf_mask=getattr(hparams.model, "condition_with_hf_mask", False),
            condition_with_cutoff_embedding=getattr(hparams.model, "condition_with_cutoff_embedding", False),
            target_type=getattr(hparams.model, "target_type", "mel_highband_residual"),
            residual_noise_scale=getattr(hparams.model, "residual_noise_scale", 1.0),
            seam_smoothing_enabled=getattr(hparams.model, "seam_smoothing_enabled", False),
            seam_smoothing_kernel_size=getattr(hparams.model, "seam_smoothing_kernel_size", 3),
            seam_smoothing_bins=getattr(hparams.model, "seam_smoothing_bins", 4),
            use_hf_frequency_weight=getattr(hparams.train, "use_hf_frequency_weight", False),
            hf_weight_min=getattr(hparams.train, "hf_weight_min", 1.0),
            hf_weight_max=getattr(hparams.train, "hf_weight_max", 1.5),
            hf_weight_mode=getattr(hparams.train, "hf_weight_mode", "linear"),
            hf_weight_cap_hz_above_cutoff=getattr(hparams.train, "hf_weight_cap_hz_above_cutoff", 4000.0),
            loss_hf_l1_weight=getattr(hparams.train, "loss_hf_l1_weight", 0.0),
            edge_continuity_loss_weight=getattr(hparams.train, "edge_continuity_loss_weight", 0.0),
            edge_continuity_bins=getattr(hparams.train, "edge_continuity_bins", 4),
            use_freq_prediction_loss=getattr(hparams.train, "use_freq_prediction_loss", False),
            freq_prediction_loss_weight=getattr(hparams.train, "freq_prediction_loss_weight", 0.0),
            use_freq_gradient_loss=getattr(hparams.train, "use_freq_gradient_loss", False),
            freq_gradient_loss_weight=getattr(hparams.train, "freq_gradient_loss_weight", 0.0),
            melconv_bridge_l1_weight=getattr(hparams.train, "melconv_bridge_l1_weight", 0.0),
            output_mel_adapter_l1_weight=getattr(hparams.train, "output_mel_adapter_l1_weight", 0.0),
        )
        total_params, trainable_params = count_parameters(cfm_wrapper)
        cbt_params = count_named_parameters(cfm_wrapper, "cbt_bridge")
        print(f"Total params: {total_params / 1e6:.2f} M")
        print(f"Trainable params: {trainable_params / 1e6:.2f} M")
        print(f"LightweightCBTBridge params: {cbt_params} ({cbt_params / 1e6:.4f} M)")
        print_cbt_init_sanity(model)
        print_active_high_gate_sanity(model)

        resume_path = None
        if hparams.runtime.show_model_summary:
            summary(cfm_wrapper)

        accelerate_kwargs = {}
        if getattr(hparams.train, "mixed_precision", None):
            accelerate_kwargs["mixed_precision"] = hparams.train.mixed_precision

        trainer = FLowHighTrainer(
            cfm_wrapper=cfm_wrapper,
            batch_size=hparams.train.batchsize,
            dataset=train_dataset,
            validset=valid_dataset,
            num_train_steps=hparams.train.num_train_steps,
            num_warmup_steps=hparams.train.n_warmup_steps,
            num_epochs=hparams.train.num_epochs,
            lr=hparams.train.lr,
            bridge_lr=getattr(hparams.train, "bridge_lr", hparams.train.lr),
            lite_transformer_lr=getattr(hparams.train, "lite_transformer_lr", hparams.train.lr),
            output_adapter_lr=getattr(hparams.train, "output_adapter_lr", hparams.train.lr),
            initial_lr=hparams.train.initial_lr,
            wd=hparams.train.weight_decay,
            max_grad_norm=hparams.train.max_grad_norm,
            grad_accum_every=hparams.train.grad_accum_every,
            log_every=hparams.logging.log_every,
            save_results_every=hparams.train.save_results_every,
            save_model_every_epochs=hparams.checkpoint.save_model_every_epochs,
            results_folder=str(model_dir),
            random_split_seed=hparams.data.random_split_seed,
            original_sampling_rate=hparams.data.samplingrate,
            downsampling=hparams.data.downsampling_method,
            valid_prepare=True,
            split_batches=hparams.train.split_batches,
            drop_last=hparams.train.drop_last,
            force_clear_prev_results=hparams.train.force_clear_prev_results,
            find_unused_parameters=hparams.train.find_unused_parameters,
            accelerate_kwargs=accelerate_kwargs,
            sampling_rates=sampling_rates,
            cfm_method=hparams.model.cfm_path,
            weighted_loss=hparams.train.weighted_loss,
            model_name=hparams.logging.run_name,
            tensorboard_logger=logger,
            validation_num_samples=hparams.validation.validation_num_samples,
            validation_time_steps=hparams.validation.validation_time_steps,
            fixed_validation_sample_rates=hparams.validation.fixed_validation_sample_rates,
            validate_on_epoch_end=hparams.validation.validate_on_epoch_end,
            val_step_interval=hparams.validation.val_step_interval,
            cache_validation_batch=hparams.validation.cache_validation_batch,
            validation_save_audio=hparams.validation.validation_save_audio,
            validation_save_audio_num=hparams.validation.validation_save_audio_num,
            validation_batch_size=hparams.validation.validation_batch_size,
            validation_audio_dirname=hparams.validation.validation_audio_dirname,
            save_best_checkpoint=hparams.checkpoint.save_best_checkpoint,
            best_metric_name=hparams.checkpoint.best_metric_name,
            best_metric_mode=hparams.checkpoint.best_metric_mode,
            best_checkpoint_name=hparams.checkpoint.best_checkpoint_name,
            best_checkpoint_template=hparams.checkpoint.best_checkpoint_template,
            epoch_checkpoint_template=hparams.checkpoint.epoch_checkpoint_template,
            final_checkpoint_template=hparams.checkpoint.final_checkpoint_template,
            tracker_project_name=hparams.logging.tracker_project_name,
            metrics_txt_path=str(metrics_txt_path),
            tensorboard_train_keys=hparams.logging.tensorboard_train_keys,
            tensorboard_valid_keys=hparams.logging.tensorboard_valid_keys,
            txt_epoch_keys=hparams.logging.txt_epoch_keys,
            scheduler_type=hparams.train.scheduler_type,
            input_sample_rates=hparams.train.input_sample_rates,
            sr_sampling_probs=hparams.train.sr_sampling_probs,
            compute_lsd_metrics=hparams.validation.compute_lsd_metrics,
            compute_visqol=hparams.validation.compute_visqol,
            visqol_bin=hparams.validation.visqol_bin,
        )

        trainer.train(mid_ckpt=None)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

