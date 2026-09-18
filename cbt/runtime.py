"""Small CBT inference adapter shared by infer.py and test.py."""

from pathlib import Path

import numpy as np
import torch

from .model_loader import build_cbt_model


def load_model(checkpoint: Path, vocoder_checkpoint: Path):
    from .config_highband_cbtbridge_full import config

    root = Path(__file__).resolve().parents[1]
    config.model.vocoderpath = str(vocoder_checkpoint.resolve())
    config.model.vocoderconfigpath = str(root / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json")
    if not torch.cuda.is_available():
        raise RuntimeError("This CBT/BigVGAN implementation requires CUDA")
    model, _ = build_cbt_model(config, checkpoint.resolve(), torch.device("cuda"))
    return model, config


@torch.inference_mode()
def predict(model, config, lr_up: np.ndarray, input_sr: int, seed: int) -> np.ndarray:
    codec = model.flowhigh.audio_enc_dec
    waveform = torch.from_numpy(np.ascontiguousarray(lr_up, dtype=np.float32)).unsqueeze(0).cuda()
    mel_up = codec.encode(waveform)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    noise = torch.randn(mel_up.shape, device=mel_up.device, dtype=mel_up.dtype, generator=generator)
    mel_pred = model.sample(
        cond=mel_up, time_steps=1, decode_to_audio=False,
        random_sr=int(input_sr), input_sampling_rate=int(input_sr),
        cfm_method=config.model.cfm_path, mel_pp=False,
        validation_generator=generator, initial_noise=noise,
    )
    return codec.decode(mel_pred).detach().float().cpu().reshape(-1).numpy()
