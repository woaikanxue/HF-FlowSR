# HF-FlowSR: CBTBridge-Full

Official research implementation of **HF-FlowSR with CBTBridge-Full** for speech audio super-resolution.

This repository provides the model architecture, training and inference pipelines, and evaluation utilities used for CBTBridge-Full. The implementation supports bandwidth extension from multiple low-sampling-rate inputs to 48 kHz speech.

> **Pretrained model:** [HF-FlowSR on Hugging Face](https://huggingface.co/zhuow1343/HF-FlowSR)

## Repository Layout

```text
src/hf_flowsr/
  configs/             CBTBridge-Full architecture and training configuration
  model.py             acoustic model, CBTBridge, and flow-matching wrapper
  modules.py           transformer and network components
  train_impl.py        training pipeline
  trainer.py           optimization and validation
  inference.py         shared prediction API
  protocol.py          degradation and evaluation utilities
  cli/                 training, inference, and evaluation commands
  third_party/bigvgan/ BigVGAN runtime and associated licenses

docs/                  model and evaluation documentation
data/                  expected dataset layout
checkpoints/           checkpoint directory
tests/                 regression and protocol tests
```

## Installation

The codebase is developed with **Python 3.10** and a CUDA-capable PyTorch environment.

Install the package with:

```bash
python -m pip install -e .
```

Please install a compatible CUDA-enabled PyTorch version according to your local environment.

HF-FlowSR uses **BigVGAN** for waveform reconstruction. A compatible 48 kHz BigVGAN checkpoint is required for waveform inference and evaluation.

The pretrained HF-FlowSR checkpoint is available from the [HF-FlowSR Hugging Face repository](https://huggingface.co/zhuow1343/HF-FlowSR).

## Training

Prepare clean **48 kHz speech** organized in speaker subdirectories.

The default audio format is FLAC. WAV files can be used by specifying:

```bash
--audio-extension .wav
```

Model architecture and optimization settings are defined in:

```text
src/hf_flowsr/configs/cbt_full.py
```

Example:

```bash
hf-flowsr-train \
  --data-root /path/to/train_audio \
  --vocoder-checkpoint /path/to/bigvgan_checkpoint \
  --output-dir runs/cbt_full
```

This command trains CBTBridge-Full from scratch and stores checkpoints and TensorBoard logs under the specified output directory.

The number of training epochs can be changed using:

```bash
--epochs
```

## Inference

HF-FlowSR supports speech inputs sampled at:

* 8 kHz
* 12 kHz
* 16 kHz
* 24 kHz

The reconstructed waveform is generated at **48 kHz**.

Example:

```bash
hf-flowsr-infer \
  --input low.wav \
  --output restored.wav \
  --checkpoint /path/to/hf_flowsr_checkpoint \
  --vocoder-checkpoint /path/to/bigvgan_checkpoint
```

Optional waveform post-processing can be enabled with:

```bash
--waveform-pp
```

This applies the low-frequency waveform anchoring procedure used by the evaluation pipeline.

## Evaluation

The evaluation pipeline accepts clean 48 kHz reference speech and automatically generates bandwidth-limited inputs at:

```text
8 kHz
12 kHz
16 kHz
24 kHz
```

For each input bandwidth, the model performs speech super-resolution and evaluates the reconstructed outputs using the unified evaluation protocol implemented in this repository.

Example:

```bash
hf-flowsr-eval \
  --audio-root /path/to/test_audio \
  --checkpoint /path/to/hf_flowsr_checkpoint \
  --vocoder-checkpoint /path/to/bigvgan_checkpoint \
  --output-dir results/cbt_full
```

For a quick pipeline check, use:

```bash
--limit 1
```

The evaluation pipeline produces:

```text
per_utterance.csv
summary.csv
```

along with reconstructed waveform caches when enabled.

Detailed evaluation settings and metric definitions are provided in:

```text
docs/EVALUATION_PROTOCOL.md
```

Users who wish to reproduce reported benchmark results should use the corresponding dataset split and evaluation settings described in the paper.

## Pretrained Models

Pretrained **HF-FlowSR / CBTBridge-Full** model weights are available on Hugging Face:

**[Download HF-FlowSR pretrained models](https://huggingface.co/zhuow1343/HF-FlowSR)**

Please refer to the model repository for the available checkpoint files and download instructions.

## Development Checks

Run the regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Third-Party Components

This repository includes components derived from the following open-source projects:

* [NVIDIA BigVGAN](https://github.com/NVIDIA/BigVGAN)
* [alias-free-torch](https://github.com/junjun3518/alias-free-torch)

The corresponding third-party licenses are included with their source code.

Please refer to the individual license files for details.

## Citation

If you find this repository useful for your research, please consider citing the corresponding HF-FlowSR paper.

Citation information will be added following publication.
