# CBTBridge-Full model card

## Model

CBTBridge-Full is the high-band residual flow model from the local HF-FlowSR
research workspace. Its implementation combines a cutoff-conditioned acoustic
model, CBTBridge, OutputMelAdapter, and a 48 kHz BigVGAN waveform decoder.
The frozen architecture parameters are in
[`../src/hf_flowsr/configs/cbt_full.py`](../src/hf_flowsr/configs/cbt_full.py).

## Inputs and outputs

- Input: 8, 12, 16, or 24 kHz speech, supplied as WAV/FLAC for inference.
- Output: 48 kHz float32 WAV. Optional low-band waveform anchoring can be
  selected with `--waveform-pp`.
- Training and test references: clean 48 kHz speech.

## Weights and evaluation

The source package contains no model or vocoder weights. The evaluator uses
the shared `Review_one` signal and LSD functions, documented in
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md). The official frozen test
manifest was unavailable in the local source workspace; results on a user
supplied directory must identify that directory and must not be presented as
the exact paper benchmark.

## Scope

This release contains one CBTBridge-Full model. It does not include the
FlowHigh baseline, ablation variants, or external comparison models.
