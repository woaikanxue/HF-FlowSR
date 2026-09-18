from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.signal
import torch


# This identifier is part of every cache key and result row. Any change to the
# signal path or metric definition must increment it.
PROTOCOL_VERSION = "hf-flowsr-formal-v1.1.0-rc1"
METRIC_VERSION = "lsd-log10power-nfft2048-hop512-fastwave-hfbin-v2"
NORMALIZATION_VERSION = "explicit-none-or-training-peak-v2"
ALIGNMENT_VERSION = "explicit-policy-none-sharedlr-permodel-v2"
DEGRADATION_VERSION = "cheby1-o8-rp005-cutoff098nyq-resamplepoly-v1"
WAVEFORM_PRECISION_VERSION = "float32-npy-metrics-int16-preview-only-v1"
POSTPROCESS_VERSION = "complex-stft-lowband-anchor-nfft2048-hop480-v2"
MEL_FRONTEND_RECORD_VERSION = "live-audio-enc-dec-signature-v1"
SMOOTHING_VERSION = "mel-seam-avgpool-k3-radius4-v1"


@dataclass(frozen=True)
class FormalProtocol:
    target_sr: int = 48000
    metric_n_fft: int = 2048
    metric_win_length: int = 2048
    metric_hop_length: int = 512
    metric_center: bool = True
    metric_window: str = "hann"
    metric_power_floor: float = 1e-8
    postprocess_n_fft: int = 2048
    postprocess_win_length: int = 2048
    postprocess_hop_length: int = 480
    postprocess_center: bool = True
    postprocess_window: str = "hann"
    mel_n_fft: int = 2048
    mel_win_length: int = 2048
    mel_hop_length: int = 480
    mel_n_mels: int = 256
    mel_fmin: float = 20.0
    mel_fmax: float = 24000.0
    diagnostic_peak_target: float = 0.95
    alignment_max_ms: float = 50.0
    alignment_lowpass_ratio: float = 0.95
    degradation_filter_order: int = 8
    degradation_ripple_db: float = 0.05
    degradation_cutoff_ratio: float = 0.98
    smoothing_kernel_size: int = 3
    smoothing_radius_bins: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "metric_version": METRIC_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "alignment_version": ALIGNMENT_VERSION,
            "degradation_version": DEGRADATION_VERSION,
            "waveform_precision_version": WAVEFORM_PRECISION_VERSION,
            "postprocess_version": POSTPROCESS_VERSION,
            "mel_frontend_record_version": MEL_FRONTEND_RECORD_VERSION,
            "smoothing_version": SMOOTHING_VERSION,
            **asdict(self),
            "lsd_definition": (
                "mean_frame(sqrt(mean_frequency((log10(max(|STFT(pred)|^2,floor)) - "
                "log10(max(|STFT(ref)|^2,floor)))^2)))"
            ),
            "full_lf_hf_lsd_protocol": "FastWave-compatible protocol",
            "hf_bin_definition": "int((metric_n_fft // 2 + 1) * input_sr / target_sr)",
            "full_definition": "all RFFT bins",
            "lf_definition": "diff[:hf_bin, :]",
            "hf_definition": "diff[hf_bin:, :]",
            "pooled_and_diagnostic_band_definition": "physical-frequency masks (unchanged)",
            "default_normalization_policy": "none",
            "normalization_policies": {
                "none": (
                    "preserve decoded HR amplitude and do not independently normalize LR-up; "
                    "no peak/RMS normalization"
                ),
                "training_peak": (
                    "normalize HR per utterance to peak 1 before degradation, then independently "
                    "normalize LR-up to peak 1; no RMS normalization"
                ),
            },
            "precision_definition": (
                "all metrics consume in-memory/cached float32 .npy; int16 WAV is preview-only"
            ),
        }


PROTOCOL = FormalProtocol()

PRIMARY_BANDS = (
    ("0_4k", 0.0, 4000.0),
    ("4_8k", 4000.0, 8000.0),
    ("8_12k", 8000.0, 12000.0),
    ("12_24k", 12000.0, 24000.0),
)

DIAGNOSTIC_BANDS = (
    ("4_6k", 4000.0, 6000.0),
    ("6_8k", 6000.0, 8000.0),
    ("12_16k", 12000.0, 16000.0),
    ("16_24k", 16000.0, 24000.0),
)
BANDS = PRIMARY_BANDS + DIAGNOSTIC_BANDS


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_save_npy(path: str | Path, waveform: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=str(path.parent))
    os.close(fd)
    try:
        np.save(tmp_name, np.asarray(waveform, dtype=np.float32), allow_pickle=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def environment_record() -> dict[str, Any]:
    cuda_name = None
    if torch.cuda.is_available():
        try:
            cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            cuda_name = "available-unresolved"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_device": cuda_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def resolve_inference_seed(base_seed: int, audio_sha256: str, input_sr: int) -> int:
    """Resolve a paired inference seed independent of method, NFE and processing."""

    identity = canonical_json({
        "base_seed": int(base_seed),
        "audio_sha256": str(audio_sha256).lower(),
        "input_sr": int(input_sr),
    }).encode("utf-8")
    # Torch accepts signed 64-bit seeds. Keep the result positive and stable.
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def mono_float32(waveform: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(waveform):
        array = waveform.detach().cpu().numpy()
    else:
        array = np.asarray(waveform)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim == 2:
        # Treat the smaller dimension as channels. Formal VCTK files are mono,
        # but this keeps conversion deterministic if stereo slips in.
        channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
        return np.ascontiguousarray(array.mean(axis=channel_axis, dtype=np.float32))
    return np.ascontiguousarray(array.reshape(-1))


def normalize_reference(reference: np.ndarray, target_peak: float = 0.95) -> tuple[np.ndarray, dict[str, float]]:
    """Diagnostic peak normalization retained for comparison with the rc1 candidate."""
    reference = mono_float32(reference)
    if reference.size == 0:
        raise ValueError("empty reference waveform")
    peak_before = float(np.max(np.abs(reference)))
    if not math.isfinite(peak_before) or peak_before <= 1e-8:
        raise ValueError(f"invalid/silent reference peak: {peak_before}")
    gain = float(target_peak) / peak_before
    normalized = np.ascontiguousarray(reference * np.float32(gain), dtype=np.float32)
    return normalized, {
        "reference_peak_before": peak_before,
        "reference_peak_after": float(np.max(np.abs(normalized))),
        "reference_gain": gain,
    }


def resample_poly_exact(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float64)
    if int(source_sr) == int(target_sr):
        return waveform.astype(np.float32)
    factor = gcd(int(source_sr), int(target_sr))
    out = scipy.signal.resample_poly(
        waveform,
        up=int(target_sr) // factor,
        down=int(source_sr) // factor,
    )
    return np.asarray(out, dtype=np.float32)


def match_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    waveform = mono_float32(waveform)
    if target_length <= 0:
        raise ValueError(f"invalid target length: {target_length}")
    if waveform.shape[-1] < target_length:
        waveform = np.pad(waveform, (0, target_length - waveform.shape[-1]))
    else:
        waveform = waveform[:target_length]
    return np.ascontiguousarray(waveform, dtype=np.float32)


def make_lr_up(reference: np.ndarray, input_sr: int, protocol: FormalProtocol = PROTOCOL) -> tuple[np.ndarray, dict[str, Any]]:
    reference = mono_float32(reference)
    cutoff_hz = protocol.degradation_cutoff_ratio * (float(input_sr) / 2.0)
    sos = scipy.signal.cheby1(
        N=protocol.degradation_filter_order,
        rp=protocol.degradation_ripple_db,
        Wn=cutoff_hz,
        btype="low",
        fs=protocol.target_sr,
        output="sos",
    )
    filtered = scipy.signal.sosfiltfilt(sos, reference.astype(np.float64)).astype(np.float32)
    lr = resample_poly_exact(filtered, protocol.target_sr, int(input_sr))
    lr_up = resample_poly_exact(lr, int(input_sr), protocol.target_sr)
    lr_up = match_length(lr_up, reference.shape[-1])
    # No independent peak/RMS normalization is allowed here.
    return lr_up, {
        "input_sr": int(input_sr),
        "cutoff_hz": cutoff_hz,
        "lr_num_samples": int(lr.shape[-1]),
        "lr_up_peak": float(np.max(np.abs(lr_up))),
    }


def int16_wav_roundtrip(waveform: np.ndarray) -> np.ndarray:
    """Emulate the legacy save-as-int16 then normalized-WAV-load metric path."""
    waveform = mono_float32(waveform)
    pcm = (np.clip(waveform, -1.0, 1.0) * np.float32(32767.0)).astype(np.int16)
    return np.ascontiguousarray(pcm.astype(np.float32) / np.float32(32768.0))


def _alignment_view(waveform: np.ndarray, input_sr: int, protocol: FormalProtocol) -> np.ndarray:
    waveform = mono_float32(waveform).astype(np.float64)
    cutoff_hz = min(
        protocol.alignment_lowpass_ratio * float(input_sr) / 2.0,
        0.45 * float(protocol.target_sr),
    )
    sos = scipy.signal.butter(6, cutoff_hz, btype="low", fs=protocol.target_sr, output="sos")
    view = scipy.signal.sosfiltfilt(sos, waveform)
    view = view - np.mean(view)
    scale = float(np.sqrt(np.sum(view * view)))
    if scale > 1e-12:
        view = view / scale
    return view


def estimate_alignment_lag(
    prediction: np.ndarray,
    reference: np.ndarray,
    input_sr: int,
    protocol: FormalProtocol = PROTOCOL,
) -> tuple[int, float]:
    prediction = mono_float32(prediction)
    reference = mono_float32(reference)
    n = min(prediction.shape[-1], reference.shape[-1])
    if n <= 0:
        raise ValueError("cannot align empty waveforms")
    pred_view = _alignment_view(prediction[:n], input_sr, protocol)
    ref_view = _alignment_view(reference[:n], input_sr, protocol)
    correlation = scipy.signal.correlate(pred_view, ref_view, mode="full", method="fft")
    lags = scipy.signal.correlation_lags(pred_view.size, ref_view.size, mode="full")
    max_lag = int(round(protocol.alignment_max_ms * protocol.target_sr / 1000.0))
    allowed = np.abs(lags) <= max_lag
    if not bool(np.any(allowed)):
        raise RuntimeError("alignment lag window is empty")
    local_index = int(np.argmax(correlation[allowed]))
    allowed_lags = lags[allowed]
    allowed_corr = correlation[allowed]
    lag = int(allowed_lags[local_index])
    score = float(allowed_corr[local_index])
    return lag, score


def apply_fixed_lag(
    prediction: np.ndarray,
    reference: np.ndarray,
    lag: int,
    protocol: FormalProtocol = PROTOCOL,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prediction = mono_float32(prediction)
    reference = mono_float32(reference)
    if lag > 0:
        aligned_pred = prediction[lag:]
        aligned_ref = reference[: aligned_pred.shape[-1]]
    elif lag < 0:
        aligned_ref = reference[-lag:]
        aligned_pred = prediction[: aligned_ref.shape[-1]]
    else:
        n = min(prediction.shape[-1], reference.shape[-1])
        aligned_pred = prediction[:n]
        aligned_ref = reference[:n]
    n = min(aligned_pred.shape[-1], aligned_ref.shape[-1])
    if n < protocol.metric_n_fft:
        raise ValueError(f"aligned waveform too short for LSD: {n} < {protocol.metric_n_fft}")
    return (
        np.ascontiguousarray(aligned_pred[:n], dtype=np.float32),
        np.ascontiguousarray(aligned_ref[:n], dtype=np.float32),
        {
            "alignment_lag_samples": lag,
            "alignment_lag_ms": 1000.0 * lag / float(protocol.target_sr),
            "aligned_num_samples": n,
        },
    )


def align_by_policy(
    prediction: np.ndarray,
    reference: np.ndarray,
    input_sr: int,
    policy: str = "none",
    fixed_lag: int | None = None,
    protocol: FormalProtocol = PROTOCOL,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    policy = str(policy).lower()
    if policy == "none":
        lag, score, source = 0, math.nan, "none"
    elif policy == "shared_lr_xcorr":
        if fixed_lag is None:
            raise ValueError("shared_lr_xcorr requires a lag estimated from LR-up/reference")
        lag, score, source = int(fixed_lag), math.nan, "lr_up_reference"
    elif policy == "per_model_xcorr":
        lag, score = estimate_alignment_lag(prediction, reference, input_sr, protocol)
        source = "prediction_reference"
    else:
        raise ValueError(f"unsupported alignment policy: {policy}")
    aligned_pred, aligned_ref, record = apply_fixed_lag(prediction, reference, lag, protocol)
    record.update({
        "alignment_policy": policy,
        "alignment_lag_source": source,
        "alignment_score": score,
    })
    return aligned_pred, aligned_ref, record


def align_waveforms(
    prediction: np.ndarray,
    reference: np.ndarray,
    input_sr: int,
    protocol: FormalProtocol = PROTOCOL,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-compatible diagnostic alignment using per-model xcorr."""

    return align_by_policy(prediction, reference, input_sr, "per_model_xcorr", None, protocol)


def _log_power(waveform: np.ndarray, protocol: FormalProtocol) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.from_numpy(mono_float32(waveform))
    window = torch.hann_window(protocol.metric_win_length, dtype=torch.float32)
    spec = torch.stft(
        tensor,
        n_fft=protocol.metric_n_fft,
        hop_length=protocol.metric_hop_length,
        win_length=protocol.metric_win_length,
        window=window,
        center=protocol.metric_center,
        return_complex=True,
    )
    log_power = torch.log10(spec.abs().pow(2).clamp_min(protocol.metric_power_floor))
    freqs = torch.fft.rfftfreq(protocol.metric_n_fft, d=1.0 / float(protocol.target_sr))
    return log_power, freqs


def _band_lsd(diff: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return math.nan
    band = diff[mask, :]
    per_frame = torch.sqrt(torch.mean(band.pow(2), dim=0))
    return float(torch.mean(per_frame).item())


def _slice_lsd(diff: torch.Tensor, start: int | None, stop: int | None) -> float:
    band = diff[slice(start, stop), :]
    if band.shape[0] == 0:
        return math.nan
    per_frame = torch.sqrt(torch.mean(band.pow(2), dim=0))
    return float(torch.mean(per_frame).item())


def fastwave_hf_bin(input_sr: int, protocol: FormalProtocol = PROTOCOL) -> int:
    hf_bin = int((protocol.metric_n_fft // 2 + 1) * int(input_sr) / protocol.target_sr)
    num_rfft_bins = protocol.metric_n_fft // 2 + 1
    if not 0 < hf_bin < num_rfft_bins:
        raise ValueError(
            f"FastWave hf_bin must split the RFFT bins, got {hf_bin} for input_sr={input_sr}"
        )
    return hf_bin


def compute_lsd_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    input_sr: int,
    alignment_policy: str = "none",
    fixed_lag: int | None = None,
    protocol: FormalProtocol = PROTOCOL,
    mask_softness_hz: float = 200.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    aligned_pred, aligned_ref, alignment = align_by_policy(
        prediction, reference, input_sr, alignment_policy, fixed_lag, protocol
    )
    pred_logp, freqs = _log_power(aligned_pred, protocol)
    ref_logp, _ = _log_power(aligned_ref, protocol)
    frames = min(pred_logp.shape[-1], ref_logp.shape[-1])
    diff = pred_logp[:, :frames] - ref_logp[:, :frames]
    cutoff_hz = float(input_sr) / 2.0
    hf_bin = fastwave_hf_bin(input_sr, protocol)
    metrics: dict[str, float] = {
        "lsd": _band_lsd(diff, torch.ones_like(freqs, dtype=torch.bool)),
        "lsd_lf": _slice_lsd(diff, None, hf_bin),
        "lsd_hf": _slice_lsd(diff, hf_bin, None),
    }
    for name, low, high in BANDS:
        include_high = high >= protocol.target_sr / 2.0
        mask = (freqs >= low) & ((freqs <= high) if include_high else (freqs < high))
        metrics[f"lsd_{name}"] = _band_lsd(diff, mask)
    for width in (250.0, 500.0, 1000.0):
        low = max(0.0, cutoff_hz - width)
        high = min(protocol.target_sr / 2.0, cutoff_hz + width)
        metrics[f"edge_lsd_{int(width)}hz"] = _band_lsd(diff, (freqs >= low) & (freqs <= high))
    local_masks, local_metadata = cutoff_local_frequency_masks(
        freqs, input_sr, mask_softness_hz, protocol
    )
    metrics.update({name: _band_lsd(diff, mask) for name, mask in local_masks.items()})
    alignment["cutoff_local_metric_metadata"] = local_metadata
    return metrics, alignment


def cutoff_local_frequency_masks(
    freqs: torch.Tensor,
    input_sr: int,
    mask_softness_hz: float,
    protocol: FormalProtocol = PROTOCOL,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if float(mask_softness_hz) <= 0:
        raise ValueError("mask_softness_hz must be positive")
    cutoff_hz = float(input_sr) / 2.0
    nyquist_hz = float(protocol.target_sr) / 2.0
    hf_mask = torch.sigmoid((freqs - cutoff_hz) / float(mask_softness_hz))
    masks = {
        "lsd_transition": (hf_mask > 0.1) & (hf_mask < 0.9),
        "lsd_nearhf_1k": (freqs >= cutoff_hz) & (freqs < min(cutoff_hz + 1000.0, nyquist_hz)),
        "lsd_nearhf_2k": (freqs >= cutoff_hz) & (freqs < min(cutoff_hz + 2000.0, nyquist_hz)),
        "lsd_farhf": freqs >= cutoff_hz + 4000.0,
    }
    metadata = {
        "cutoff_hz": cutoff_hz,
        "mask_softness_hz": float(mask_softness_hz),
        "stft_frequency_formula": "f_k = k * target_sr / n_fft",
        "transition_definition": "0.1 < sigmoid((f-fc)/softness_hz) < 0.9",
        "frequency_bin_counts": {name: int(mask.sum().item()) for name, mask in masks.items()},
        "empty_frequency_bands": [name for name, mask in masks.items() if not bool(mask.any())],
    }
    return masks, metadata


def compute_cutoff_local_lsd_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    input_sr: int,
    *,
    mask_softness_hz: float,
    alignment_policy: str = "none",
    fixed_lag: int | None = None,
    protocol: FormalProtocol = PROTOCOL,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Physical-frequency cutoff diagnostics using the model's sigmoid HF mask."""
    aligned_pred, aligned_ref, alignment = align_by_policy(
        prediction, reference, input_sr, alignment_policy, fixed_lag, protocol
    )
    pred_logp, freqs = _log_power(aligned_pred, protocol)
    ref_logp, _ = _log_power(aligned_ref, protocol)
    frames = min(pred_logp.shape[-1], ref_logp.shape[-1])
    diff = pred_logp[:, :frames] - ref_logp[:, :frames]
    masks, local_metadata = cutoff_local_frequency_masks(
        freqs, input_sr, mask_softness_hz, protocol
    )
    metrics = {name: _band_lsd(diff, mask) for name, mask in masks.items()}
    metadata = {
        **alignment,
        **local_metadata,
    }
    return metrics, metadata


def waveform_lowband_anchor(
    prediction: np.ndarray | torch.Tensor,
    lr_up: np.ndarray | torch.Tensor,
    input_sr: int,
    protocol: FormalProtocol = PROTOCOL,
) -> torch.Tensor:
    pred = torch.as_tensor(prediction, dtype=torch.float32).reshape(-1)
    up = torch.as_tensor(lr_up, dtype=torch.float32, device=pred.device).reshape(-1)
    n = min(pred.numel(), up.numel())
    pred = pred[:n]
    up = up[:n]
    window = torch.hann_window(protocol.postprocess_win_length, device=pred.device, dtype=pred.dtype)
    pred_spec = torch.stft(
        pred, protocol.postprocess_n_fft, protocol.postprocess_hop_length,
        protocol.postprocess_win_length, window=window,
        center=protocol.postprocess_center, return_complex=True,
    )
    up_spec = torch.stft(
        up, protocol.postprocess_n_fft, protocol.postprocess_hop_length,
        protocol.postprocess_win_length, window=window,
        center=protocol.postprocess_center, return_complex=True,
    )
    frames = min(pred_spec.shape[-1], up_spec.shape[-1])
    pred_spec = pred_spec[:, :frames]
    up_spec = up_spec[:, :frames]
    freqs = torch.fft.rfftfreq(protocol.postprocess_n_fft, d=1.0 / protocol.target_sr).to(pred.device)
    mixed = pred_spec.clone()
    mixed[freqs <= float(input_sr) / 2.0, :] = up_spec[freqs <= float(input_sr) / 2.0, :]
    return torch.istft(
        mixed,
        protocol.postprocess_n_fft,
        protocol.postprocess_hop_length,
        protocol.postprocess_win_length,
        window=window,
        center=protocol.postprocess_center,
        length=n,
    )


def smooth_mel_cutoff(
    mel: torch.Tensor,
    input_sr: int,
    f_min: float,
    f_max: float,
    n_mels: int | None = None,
    protocol: FormalProtocol = PROTOCOL,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if mel.ndim != 3:
        raise ValueError(f"expected 3D mel tensor, got {tuple(mel.shape)}")
    # Formal model outputs use [B,T,M]. An explicit model n_mels removes all
    # ambiguity for short utterances where T can be smaller than M.
    n_mels = int(mel.shape[-1] if n_mels is None else n_mels)
    if mel.shape[-1] == n_mels:
        freq_last = True
    elif mel.shape[1] == n_mels:
        freq_last = False
    else:
        raise ValueError(f"n_mels={n_mels} is incompatible with mel shape {tuple(mel.shape)}")
    work = mel if freq_last else mel.transpose(1, 2)
    mel_min = 2595.0 * math.log10(1.0 + float(f_min) / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + min(float(f_max), protocol.target_sr / 2.0) / 700.0)
    centers = 700.0 * (10.0 ** (np.linspace(mel_min, mel_max, n_mels) / 2595.0) - 1.0)
    cutoff_bin = int(np.searchsorted(centers, float(input_sr) / 2.0, side="left"))
    lo = max(0, cutoff_bin - protocol.smoothing_radius_bins)
    hi = min(n_mels, cutoff_bin + protocol.smoothing_radius_bins)
    seam_mask = torch.zeros((1, 1, n_mels), device=work.device, dtype=work.dtype)
    seam_mask[..., lo:hi] = 1.0
    flat = work.reshape(-1, 1, n_mels)
    smooth = torch.nn.functional.avg_pool1d(
        flat,
        kernel_size=protocol.smoothing_kernel_size,
        stride=1,
        padding=protocol.smoothing_kernel_size // 2,
    ).reshape_as(work)
    out = work * (1.0 - seam_mask) + smooth * seam_mask
    result = out if freq_last else out.transpose(1, 2)
    metadata = {
        "common_smoothing_applied": True,
        "actual_n_mels": n_mels,
        "cutoff_bin": cutoff_bin,
        "smoothing_bin_start": lo,
        "smoothing_bin_end_exclusive": hi,
        "kernel_size": protocol.smoothing_kernel_size,
        "radius_bins": protocol.smoothing_radius_bins,
    }
    return (result, metadata) if return_metadata else result


def mel_smoothing_metadata(
    input_sr: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    applied: bool,
    protocol: FormalProtocol = PROTOCOL,
) -> dict[str, Any]:
    mel_min = 2595.0 * math.log10(1.0 + float(f_min) / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + min(float(f_max), protocol.target_sr / 2.0) / 700.0)
    centers = 700.0 * (10.0 ** (np.linspace(mel_min, mel_max, int(n_mels)) / 2595.0) - 1.0)
    cutoff_bin = int(np.searchsorted(centers, float(input_sr) / 2.0, side="left"))
    return {
        "common_smoothing_applied": bool(applied),
        "actual_n_mels": int(n_mels),
        "cutoff_bin": cutoff_bin,
        "smoothing_bin_start": max(0, cutoff_bin - protocol.smoothing_radius_bins),
        "smoothing_bin_end_exclusive": min(int(n_mels), cutoff_bin + protocol.smoothing_radius_bins),
        "kernel_size": protocol.smoothing_kernel_size,
        "radius_bins": protocol.smoothing_radius_bins,
    }


def cache_key(payload: dict[str, Any]) -> str:
    required = {
        "protocol_version",
        "method",
        "checkpoint_sha256",
        "config_sha256",
        "model_implementation_sha256",
        "repository_source_sha256",
        "vocoder_state_sha256",
        "mel_frontend_signature_sha256",
        "resolved_seed",
        "nfe",
        "mask_mode_effective",
        "mask_signature",
        "native_mel_pp_effective",
        "common_smoothing",
        "waveform_post_processing",
        "metric_version",
        "alignment_version",
        "degradation_version",
        "normalization_policy",
        "formal_code_sha256",
        "audio_sha256",
        "input_sr",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"cache key payload missing fields: {missing}")
    return sha256_json(payload)


def cache_payload(
    *,
    method: str,
    checkpoint_sha256: str,
    config_sha256: str,
    model_implementation_sha256: str,
    repository_source_sha256: str,
    vocoder_state_sha256: str,
    mel_frontend_signature_sha256: str,
    resolved_seed: int,
    nfe: int,
    mask_mode_effective: str,
    mask_signature: dict[str, Any] | str,
    native_mel_pp_effective: bool | str,
    common_smoothing: bool,
    waveform_post_processing: bool,
    normalization_policy: str,
    formal_code_sha256: str,
    audio_sha256: str,
    input_sr: int,
    diagnostic_compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "metric_version": METRIC_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "alignment_version": ALIGNMENT_VERSION,
        "degradation_version": DEGRADATION_VERSION,
        "waveform_precision_version": WAVEFORM_PRECISION_VERSION,
        "postprocess_version": POSTPROCESS_VERSION,
        "smoothing_version": SMOOTHING_VERSION,
        "method": method,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "model_implementation_sha256": model_implementation_sha256,
        "repository_source_sha256": repository_source_sha256,
        "vocoder_state_sha256": vocoder_state_sha256,
        "mel_frontend_signature_sha256": mel_frontend_signature_sha256,
        "resolved_seed": int(resolved_seed),
        "nfe": int(nfe),
        "mask_mode_effective": mask_mode_effective,
        "mask_signature": mask_signature,
        "native_mel_pp_effective": native_mel_pp_effective,
        "common_smoothing": bool(common_smoothing),
        "common_smoothing_version": SMOOTHING_VERSION,
        "waveform_post_processing": bool(waveform_post_processing),
        "normalization_policy": str(normalization_policy),
        "metric_version": METRIC_VERSION,
        "formal_code_sha256": formal_code_sha256,
        "audio_sha256": audio_sha256,
        "input_sr": int(input_sr),
        "diagnostic_compatibility": diagnostic_compatibility or {"profile": "formal"},
    }
