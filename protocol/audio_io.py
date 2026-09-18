from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile


SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".flac"})


def _soundfile() -> Any:
    try:
        return importlib.import_module("soundfile")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FLAC input requires the 'soundfile' package (PySoundFile/libsndfile) "
            "in the formal-evaluation environment"
        ) from exc


def decoder_environment_record() -> dict[str, str | None]:
    try:
        module = _soundfile()
    except RuntimeError:
        return {"soundfile": None, "libsndfile": None}
    return {
        "soundfile": str(getattr(module, "__version__", "unknown")),
        "libsndfile": str(getattr(module, "__libsndfile_version__", "unknown")),
    }


def read_reference_pcm(path: Path) -> tuple[int, np.ndarray]:
    """Decode WAV/FLAC without resampling or normalization.

    WAV keeps the original formal loader's explicit integer-to-float mapping.
    FLAC is decoded by libsndfile directly to the same normalized float32 PCM
    convention. Channel folding, resampling, and joint normalization remain the
    responsibility of the formal protocol.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        sample_rate, waveform = scipy.io.wavfile.read(path)
        if np.issubdtype(waveform.dtype, np.integer):
            if waveform.dtype == np.uint8:
                waveform = (waveform.astype(np.float32) - 128.0) / 128.0
            else:
                info = np.iinfo(waveform.dtype)
                waveform = waveform.astype(np.float32) / float(max(abs(info.min), abs(info.max)))
        return int(sample_rate), np.asarray(waveform)
    if suffix == ".flac":
        waveform, sample_rate = _soundfile().read(
            str(path), dtype="float32", always_2d=False
        )
        return int(sample_rate), np.asarray(waveform, dtype=np.float32)
    raise ValueError(f"formal PCM loader accepts WAV/FLAC only: {path}")


def probe_sample_rate(path: Path) -> int:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        sample_rate, _ = scipy.io.wavfile.read(path, mmap=True)
        return int(sample_rate)
    if suffix == ".flac":
        return int(_soundfile().info(str(path)).samplerate)
    raise ValueError(f"formal PCM loader accepts WAV/FLAC only: {path}")
