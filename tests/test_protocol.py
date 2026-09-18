"""Small regression checks for the exported Review_one signal path."""

import unittest

import numpy as np

from hf_flowsr.protocol import compute_lsd_metrics, make_lr_up, waveform_lowband_anchor


class ProtocolTest(unittest.TestCase):
    def test_degradation_and_metrics_for_all_rates(self):
        t = np.arange(48000, dtype=np.float32) / 48000.0
        reference = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        for input_sr in (8000, 12000, 16000, 24000):
            with self.subTest(input_sr=input_sr):
                lr_up, _ = make_lr_up(reference, input_sr)
                self.assertEqual(lr_up.shape, reference.shape)
                self.assertEqual(lr_up.dtype, np.float32)
                metrics, _ = compute_lsd_metrics(lr_up, reference, input_sr)
                self.assertTrue(np.isfinite(metrics["lsd"]))
                self.assertTrue(np.isfinite(metrics["lsd_hf"]))
                anchored = waveform_lowband_anchor(lr_up, lr_up, input_sr)
                self.assertEqual(anchored.numel(), reference.size)


if __name__ == "__main__":
    unittest.main()
