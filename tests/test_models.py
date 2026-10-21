import unittest

import torch

from models.crossvivit import RoCrossViViT
from models.lstm_att_sat import Solar_Predictor


class InferenceBehaviorTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_crossvivit_unmasked_patch_positions_and_determinism(self):
        model = RoCrossViViT(
            image_size=[16, 16],
            patch_size=[8, 8],
            dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_ratio=2,
            ctx_channels=4,
            ts_channels=1,
            ts_length=2,
            decoder_dim=8,
            decoder_depth=1,
            decoder_heads=2,
            decoder_dim_head=8,
            ctx_masking_ratio=0.5,
        ).eval()
        images = torch.randn(1, 2, 4, 16, 16)
        ground = torch.randn(1, 2, 1)
        with torch.inference_mode():
            first = model(images, ground, mask=False)[0]
            second = model(images, ground, mask=False)[0]
        self.assertEqual(tuple(first.shape), (1, 2, 1, 1))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_lstm_forecast_does_not_require_or_use_future_targets(self):
        model = Solar_Predictor(4, 120, 4, 4, 1, 4, 128, 1, 4, 12, 12, False, 1).eval()
        images = torch.randn(1, 12, 5, 120, 120)
        future = torch.randn_like(images)
        with torch.inference_mode():
            history_only = model(images, phase=2)
            with_future = model(images, future, phase=2)
        self.assertEqual(tuple(history_only.shape), (1, 12, 1))
        torch.testing.assert_close(history_only, with_future, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
