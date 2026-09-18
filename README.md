# CBTBridge audio super-resolution

This folder is a small, standalone source package extracted from the local
FLowHigh/HF-FlowSR workspace. It contains the **CBTBridge-Full** model,
training, inference, and a single-model test using the `Review_one` unified
signal and LSD protocol. It excludes other model backends, experiment logs,
datasets, and pretrained weights.

## Setup

Python 3.10 with CUDA is recommended for the pinned dependencies and this
version of CBT/BigVGAN.

```bash
pip install -r requirements.txt
```

Download or provide the CBT `best.pt` and 48 kHz BigVGAN
`g_48_00850000` weights separately. The original local repository contains
them at `Review_one/model/cbt/best.pt` and
`vocoder/BIGVGAN/checkpoint/g_48_00850000`. Weights are excluded from this
source package.

## Train

The data root should contain 48 kHz speech files in speaker folders. FLAC is
the default; add `--audio-extension .wav` for WAV data. The
original CBT training configuration is preserved in
`cbt/config_highband_cbtbridge_full.py`; `train.py` replaces its local data
and vocoder paths at runtime and starts training from scratch. The historical
split seed and model settings remain in the config.

```bash
python train.py --data-root /path/to/vctk_train --vocoder-checkpoint /path/to/g_48_00850000
```

Checkpoints and TensorBoard logs are written under `cbt/model/` and `cbt/log/`.

## Infer

Input must be a mono or stereo 8, 12, 16, or 24 kHz WAV/FLAC file.

```bash
python infer.py --input low.wav --output restored.wav \
  --checkpoint /path/to/best.pt \
  --vocoder-checkpoint /path/to/g_48_00850000
```

Add `--waveform-pp` for the optional low-band waveform anchor.

## Test

Supply a directory of clean 48 kHz WAV/FLAC files. For each file and input
rate, the script creates the degraded input, runs CBT once, derives RAW and
PP-on predictions, and writes float32 `.npy` caches plus `per_utterance.csv`.

```bash
python test.py --audio-root /path/to/wave48_test \
  --checkpoint /path/to/best.pt \
  --vocoder-checkpoint /path/to/g_48_00850000 \
  --output-dir results
```

Use `--limit 1` for a quick run. Protocol details are in
`protocol/EVALUATION_PROTOCOL.md`. The original official test manifest was not
present locally, so this package does not label arbitrary test sets as the
frozen paper benchmark.

## Source and licenses

CBT model/training code comes from the local FLowHigh/HF-FlowSR repository and
is covered by the top-level MIT `LICENSE`. The bundled BigVGAN implementation
is derived from [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN) and has its
own MIT license in `vocoder/BIGVGAN/LICENSE`. Its alias-free component is
adapted from [alias-free-torch](https://github.com/junjun3518/alias-free-torch)
under Apache 2.0; the license is bundled as
`vocoder/BIGVGAN/alias_free_torch_LICENSE.txt`.
