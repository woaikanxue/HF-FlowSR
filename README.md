# HF-FlowSR CBTBridge-Full

Research code for the CBTBridge-Full audio super-resolution model. This
repository contains one model, its training and inference code, and an
evaluation path based on the `Review_one` unified test protocol. Pretrained
weights and datasets are distributed separately.

## Repository layout

```text
src/hf_flowsr/
  configs/             CBTBridge-Full architecture and training configuration
  model.py             acoustic model, CBTBridge, and flow wrapper
  modules.py           transformer and network components
  train_impl.py        training loop assembly
  trainer.py           optimization and validation
  inference.py         shared prediction API
  protocol.py          unified degradation and LSD functions
  cli/                 train, infer, and evaluate commands
  third_party/bigvgan/ bundled vocoder runtime and licenses
docs/                  model card and evaluation protocol
data/                  expected dataset layout
checkpoints/           expected weight files
tests/                 protocol regression checks
```

## Install

Use Python 3.10 and a CUDA capable PyTorch installation. The pinned package
versions reflect the source research environment.

```bash
python -m pip install -e .
```

The CBT model checkpoint (`best.pt`) and 48 kHz BigVGAN checkpoint
(`g_48_00850000`) are required for inference and evaluation. The original
workspace stored them at `Review_one/model/cbt/best.pt` and
`vocoder/BIGVGAN/checkpoint/g_48_00850000`. Pass local paths with the commands
below; see [checkpoints/README.md](checkpoints/README.md).

## Train

Supply clean 48 kHz speech in speaker subdirectories. The default file type
is FLAC; use `--audio-extension .wav` for WAV. The model and optimization
settings are in [cbt_full.py](src/hf_flowsr/configs/cbt_full.py).

```bash
hf-flowsr-train \
  --data-root /path/to/train_audio \
  --vocoder-checkpoint /path/to/g_48_00850000 \
  --output-dir runs/cbt_full
```

The command starts CBTBridge-Full training from scratch. It writes checkpoints
and TensorBoard logs below `--output-dir`. Use `--epochs` to change the number
of epochs.

## Infer

Pass an 8, 12, 16, or 24 kHz WAV/FLAC file. The output is a 48 kHz float32
WAV file.

```bash
hf-flowsr-infer \
  --input low.wav --output restored.wav \
  --checkpoint /path/to/best.pt \
  --vocoder-checkpoint /path/to/g_48_00850000
```

Add `--waveform-pp` to apply the common low-band waveform anchor.

## Evaluate

Supply clean 48 kHz WAV/FLAC references. The evaluator creates 8, 12, 16,
and 24 kHz degraded inputs, runs CBT once for each rate, derives RAW and
PP-on outputs from the same prediction, and writes float32 caches,
`per_utterance.csv`, and a per-rate/equal-rate-macro `summary.csv`. Use
`--limit 1` for a quick check.

```bash
hf-flowsr-eval \
  --audio-root /path/to/test_audio \
  --checkpoint /path/to/best.pt \
  --vocoder-checkpoint /path/to/g_48_00850000 \
  --output-dir results/cbt_full
```

The evaluation rules are in
[EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md). The original workspace
did not contain the frozen official test manifest CSV, so this command
evaluates the files supplied by the user and does not label them as the exact
paper benchmark.

## Development checks

```bash
python -m unittest discover -s tests -v
```

## Source and licenses

The CBT code was extracted from the local HF-FlowSR research workspace and
uses the top-level [MIT license](LICENSE). The bundled BigVGAN runtime comes
from [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN), with its license in
`src/hf_flowsr/third_party/bigvgan/LICENSE`. Its alias-free component is
derived from [alias-free-torch](https://github.com/junjun3518/alias-free-torch)
under Apache 2.0; that license is included alongside the code.
