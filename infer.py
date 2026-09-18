"""Upsample one low-resolution WAV or FLAC file with CBT."""

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile

from protocol.audio_io import read_reference_pcm
from protocol.formal_protocol import mono_float32, resample_poly_exact, waveform_lowband_anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--waveform-pp", action="store_true", help="Use the common low-band anchor")
    args = parser.parse_args()

    input_sr, audio = read_reference_pcm(args.input)
    if input_sr not in (8000, 12000, 16000, 24000):
        raise ValueError("Input sample rate must be 8, 12, 16, or 24 kHz")
    lr_up = resample_poly_exact(mono_float32(audio), input_sr, 48000)
    from cbt.runtime import load_model, predict
    model, config = load_model(args.checkpoint, args.vocoder_checkpoint)
    output = predict(model, config, lr_up, input_sr, args.seed)
    output = output[:len(lr_up)] if len(output) >= len(lr_up) else np.pad(output, (0, len(lr_up) - len(output)))
    if args.waveform_pp:
        output = waveform_lowband_anchor(output, lr_up, input_sr).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(args.output, 48000, output.astype(np.float32))
    print(args.output)


if __name__ == "__main__":
    main()
