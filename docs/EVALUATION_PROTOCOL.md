# CBT test protocol (from `Review_one`)

`src/hf_flowsr/protocol.py` and `src/hf_flowsr/audio_io.py` are copied from
the source repository's `Review_one` unified test protocol. The
`hf-flowsr-eval` command uses these functions directly.

- Reference: mono float32 PCM, 48 kHz; no peak or RMS normalization.
- Input rates: 8, 12, 16, 24 kHz.
- Degradation: Chebyshev-I order 8, ripple 0.05 dB, cutoff at 0.98 of input Nyquist, followed by polyphase downsampling and upsampling.
- CBT: one flow step, native mask reconstruction, no native mel postprocessing or seam smoothing. The inference seed is derived from base seed 1234, audio SHA-256, and input rate.
- Vocoder: shared 48 kHz BigVGAN. The raw float32 prediction is saved as `.npy`.
- PP-on: low-band waveform anchor derived from the same raw prediction; FFT/window 2048, hop 480. It does not rerun CBT or BigVGAN.
- LSD: Hann STFT, FFT/window 2048, hop 512, log10 power floor `1e-8`. High-band split uses `int((2048//2+1)*input_sr/48000)`.
- Alignment: none. Metrics consume float32 waveforms.

`hf-flowsr-eval` evaluates whatever 48 kHz WAV/FLAC files the user supplies. The source
repository's frozen official manifest CSV is absent from the local `Review_one`
directory, so this standalone test is a protocol-compatible evaluation, not a
claim of reproducing the exact 2,346-file paper test set.
