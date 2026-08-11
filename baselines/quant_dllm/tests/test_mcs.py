import unittest

try:
    import torch

    from utils.mcs import apply_mcs, timestep_for_sample
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskedCalibrationSimulationTest(unittest.TestCase):
    def test_uniform_timestep_grid(self):
        self.assertEqual(
            [timestep_for_sample(index, 4) for index in range(4)],
            [0.25, 0.5, 0.75, 1.0],
        )

    def test_prefix_is_always_visible(self):
        batch = torch.arange(8).unsqueeze(0)
        noisy, probabilities = apply_mcs(
            batch,
            mask_id=99,
            sample_index=3,
            num_samples=4,
            prefix_ratio=0.25,
            seed=7,
        )
        self.assertTrue(torch.equal(noisy[:, :2], batch[:, :2]))
        self.assertTrue(torch.equal(probabilities[:, :2], torch.zeros((1, 2))))
        self.assertTrue(torch.equal(noisy[:, 2:], torch.full((1, 6), 99)))

    def test_masks_are_deterministic_per_sample(self):
        batch = torch.arange(128).unsqueeze(0)
        first, _ = apply_mcs(batch, 999, 5, 16, seed=11)
        repeated, _ = apply_mcs(batch, 999, 5, 16, seed=11)
        different, _ = apply_mcs(batch, 999, 6, 16, seed=11)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))

    def test_batch_size_must_remain_one(self):
        with self.assertRaises(ValueError):
            apply_mcs(torch.zeros((2, 8), dtype=torch.long), 99, 0, 4)


if __name__ == "__main__":
    unittest.main()
