"""Train CBTBridge from scratch on a 48 kHz VCTK-style audio directory."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--audio-extension", choices=(".flac", ".wav"), default=".flac")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if not args.vocoder_checkpoint.is_file():
        raise FileNotFoundError(args.vocoder_checkpoint)

    from cbt import train_cbtbridge_full_accelerate as training

    config = training.hparams
    config.data.data_path = str(args.data_root.resolve())
    config.data.train_path = str(args.data_root.resolve())
    config.data.audio_extension = args.audio_extension
    config.train.resume_from = None
    config.train.train_from_scratch = True
    config.model.vocoderpath = str(args.vocoder_checkpoint.resolve())
    config.model.vocoderconfigpath = str(
        Path(__file__).resolve().parent / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"
    )
    config.runtime.show_model_summary = False
    if args.epochs is not None:
        config.train.num_epochs = args.epochs
    training.main()


if __name__ == "__main__":
    main()
