# Data layout

Training expects clean 48 kHz speech in speaker subdirectories, for example:

```text
/path/to/vctk_train/
  p001/utterance_001.flac
  p001/utterance_002.flac
  p002/utterance_001.flac
```

Use `--audio-extension .wav` when training on WAV. Evaluation takes a separate
directory of clean 48 kHz WAV/FLAC files and constructs all four low-rate
inputs with the protocol in [`../docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md).
No audio is included in this repository.
