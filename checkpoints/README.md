# Checkpoints

Supply model weights separately:

- CBTBridge-Full `best.pt` for inference and evaluation.
- 48 kHz BigVGAN `g_48_00850000` for training, inference, and evaluation.

Pass both paths explicitly to the CLI. Training writes new model checkpoints
to `runs/cbt_full/checkpoints/` by default. No weights are committed.
