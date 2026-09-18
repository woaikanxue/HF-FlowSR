"""Construct CBTBridge-Full and load a model checkpoint."""
from pathlib import Path
import torch

from .model import FLowHigh, MelVoco, ConditionalFlowMatcherWrapper

ROOT = Path(__file__).resolve().parents[2]

def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_cbt_model(config, checkpoint_path, device):
    data_cfg = config.data
    model_cfg = config.model
    audio_enc_dec = MelVoco(
        n_mels=data_cfg.n_mel_channels,
        sampling_rate=data_cfg.samplingrate,
        f_max=data_cfg.mel_fmax,
        f_min=data_cfg.mel_fmin,
        n_fft=data_cfg.n_fft,
        win_length=data_cfg.win_length,
        hop_length=data_cfg.hop_length,
        vocoder=model_cfg.vocoder,
        vocoder_config=str(resolve_path(model_cfg.vocoderconfigpath)),
        vocoder_path=str(resolve_path(model_cfg.vocoderpath)),
    )
    model = FLowHigh(
        architecture=model_cfg.architecture,
        dim_in=data_cfg.n_mel_channels,
        audio_enc_dec=audio_enc_dec,
        dim=model_cfg.dim,
        depth=model_cfg.n_layers,
        dim_head=model_cfg.dim_head,
        heads=model_cfg.n_heads,
        input_channels=model_cfg.input_channels,
        condition_with_hf_mask=getattr(model_cfg, "condition_with_hf_mask", False),
        condition_with_cutoff_embedding=getattr(model_cfg, "condition_with_cutoff_embedding", False),
        use_adaln_zero=getattr(model_cfg, "use_adaln_zero", False),
        use_interleaved_melconv=getattr(model_cfg, "use_interleaved_melconv", False),
        use_melconv_bridge=getattr(model_cfg, "use_melconv_bridge", False),
        cbt_bridge_enabled=getattr(model_cfg, "cbt_bridge_enabled", False),
        cbt_hidden_dim=getattr(model_cfg, "cbt_hidden_dim", 32),
        cbt_low_groups_hz=getattr(model_cfg, "cbt_low_groups_hz", None),
        cbt_high_groups_hz=getattr(model_cfg, "cbt_high_groups_hz", None),
        cbt_use_event_gate=getattr(model_cfg, "cbt_use_event_gate", True),
        cbt_use_temporal_derivative=getattr(model_cfg, "cbt_use_temporal_derivative", True),
        cbt_use_depthwise_temporal_conv=getattr(model_cfg, "cbt_use_depthwise_temporal_conv", True),
        cbt_temporal_kernel=getattr(model_cfg, "cbt_temporal_kernel", 5),
        cbt_zero_init=getattr(model_cfg, "cbt_zero_init", True),
        cbt_init_scale=getattr(model_cfg, "cbt_init_scale", 0.0),
        cbt_dropout=getattr(model_cfg, "cbt_dropout", 0.0),
        use_output_mel_adapter=getattr(model_cfg, "use_output_mel_adapter", False),
        first_transformer_depth=getattr(model_cfg, "first_transformer_depth", 1),
        first_transformer_heads=getattr(model_cfg, "first_transformer_heads", model_cfg.n_heads),
        first_transformer_dim_head=getattr(model_cfg, "first_transformer_dim_head", model_cfg.dim_head),
        first_transformer_ff_mult=getattr(model_cfg, "first_transformer_ff_mult", 4),
        second_transformer_depth=getattr(model_cfg, "second_transformer_depth", 1),
        second_transformer_heads=getattr(model_cfg, "second_transformer_heads", 8),
        second_transformer_dim_head=getattr(model_cfg, "second_transformer_dim_head", model_cfg.dim_head),
        second_transformer_ff_mult=getattr(model_cfg, "second_transformer_ff_mult", 2),
        output_mel_adapter_channels=getattr(model_cfg, "output_mel_adapter_channels", 64),
        output_mel_adapter_blocks=getattr(model_cfg, "output_mel_adapter_blocks", 2),
        output_mel_adapter_kernel_time=getattr(model_cfg, "output_mel_adapter_kernel_time", 3),
        output_mel_adapter_kernel_freq=getattr(model_cfg, "output_mel_adapter_kernel_freq", 9),
        output_mel_adapter_zero_init=getattr(model_cfg, "output_mel_adapter_zero_init", False),
        output_mel_adapter_final_init_std=getattr(model_cfg, "output_mel_adapter_final_init_std", 1e-4),
        output_mel_adapter_scale_ramp_steps=getattr(model_cfg, "output_mel_adapter_scale_ramp_steps", 5000),
        output_mel_adapter_use_v_base=getattr(model_cfg, "output_mel_adapter_use_v_base", True),
        output_mel_adapter_use_zt=getattr(model_cfg, "output_mel_adapter_use_zt", True),
        output_mel_adapter_use_cond=getattr(model_cfg, "output_mel_adapter_use_cond", True),
        output_mel_adapter_use_mask=getattr(model_cfg, "output_mel_adapter_use_mask", True),
        output_mel_adapter_use_freq_pos=getattr(model_cfg, "output_mel_adapter_use_freq_pos", True),
        output_mel_adapter_use_cutoff_dist=getattr(model_cfg, "output_mel_adapter_use_cutoff_dist", True),
        output_mel_adapter_use_band_gates=getattr(model_cfg, "output_mel_adapter_use_band_gates", True),
        output_mel_adapter_near_hz=getattr(model_cfg, "output_mel_adapter_near_hz", 4000.0),
        output_mel_adapter_mid_hz=getattr(model_cfg, "output_mel_adapter_mid_hz", 10000.0),
        output_mel_adapter_band_gate_init=getattr(model_cfg, "output_mel_adapter_band_gate_init", 1.0),
    )
    wrapper = ConditionalFlowMatcherWrapper(
        flowhigh=model,
        cfm_method=model_cfg.cfm_path,
        torchdiffeq_ode_method=getattr(config.inference, "ode_method", "euler"),
        sigma=model_cfg.sigma,
        use_highband_residual_flow=getattr(model_cfg, "use_highband_residual_flow", False),
        highband_mask_softness_hz=getattr(model_cfg, "highband_mask_softness_hz", 500.0),
        condition_with_hf_mask=getattr(model_cfg, "condition_with_hf_mask", False),
        condition_with_cutoff_embedding=getattr(model_cfg, "condition_with_cutoff_embedding", False),
        target_type=getattr(model_cfg, "target_type", "mel_highband_residual"),
        residual_noise_scale=getattr(model_cfg, "residual_noise_scale", 1.0),
        seam_smoothing_enabled=getattr(model_cfg, "seam_smoothing_enabled", False),
        seam_smoothing_kernel_size=getattr(model_cfg, "seam_smoothing_kernel_size", 3),
        seam_smoothing_bins=getattr(model_cfg, "seam_smoothing_bins", 4),
    ).to(device)

    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    wrapper.load_state_dict(state, strict=True)
    wrapper.eval()
    wrapper.flowhigh.audio_enc_dec.eval()
    return wrapper, checkpoint_path

