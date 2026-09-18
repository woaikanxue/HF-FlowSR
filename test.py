"""Evaluate CBT on clean 48 kHz WAV/FLAC audio with Review_one metrics."""

import argparse
import csv
from pathlib import Path

import numpy as np

from protocol.audio_io import read_reference_pcm
from protocol.formal_protocol import (
    compute_lsd_metrics, make_lr_up, match_length, mono_float32,
    resolve_inference_seed, sha256_file, waveform_lowband_anchor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-srs", default="8000,12000,16000,24000")
    parser.add_argument("--limit", type=int, default=0, help="Use the first N files for a quick test")
    args = parser.parse_args()

    rates = tuple(int(x) for x in args.input_srs.split(","))
    if not rates or any(x not in (8000, 12000, 16000, 24000) for x in rates):
        raise ValueError("Input rates must be 8, 12, 16, or 24 kHz")
    root = args.audio_root.resolve()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No WAV/FLAC files under {root}")

    from cbt.runtime import load_model, predict
    model, config = load_model(args.checkpoint, args.vocoder_checkpoint)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        sr, audio = read_reference_pcm(path)
        if sr != 48000:
            raise ValueError(f"Expected 48 kHz reference: {path}")
        reference = mono_float32(audio)
        audio_hash = sha256_file(path)
        relative = path.relative_to(root).with_suffix("")
        for input_sr in rates:
            lr_up, _ = make_lr_up(reference, input_sr)
            seed = resolve_inference_seed(1234, audio_hash, input_sr)
            raw = match_length(predict(model, config, lr_up, input_sr, seed), len(reference))
            pp_on = match_length(waveform_lowband_anchor(raw, lr_up, input_sr).numpy(), len(reference))
            for condition, prediction in (("raw", raw), ("pp_on", pp_on)):
                cache = output_dir / "predictions" / condition / str(input_sr) / relative.with_suffix(".npy")
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, prediction.astype(np.float32, copy=False))
                metrics, _ = compute_lsd_metrics(prediction, reference, input_sr)
                records.append({"file": str(path.relative_to(root)), "input_sr": input_sr,
                                "seed": seed, "condition": condition, **metrics})
        print(f"Evaluated {path.relative_to(root)}")

    with (output_dir / "per_utterance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(output_dir / "per_utterance.csv")


if __name__ == "__main__":
    main()
