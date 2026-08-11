import tempfile
import unittest

import torch
import torch.nn as nn

from trajmc import common
from trajmc.calibration import apply_noise
from trajmc.compression import plan_bins, whiten_truncate
from eval.run import build_command


class LLaDABlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in common.get_backend("llada").all_suffixes:
            setattr(self, name, nn.Linear(8, 8, bias=False))


class TinyLLaDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.blocks = nn.ModuleList([LLaDABlock()])
        self.model.transformer.ff_out = nn.Linear(8, 16, bias=False)

    def get_output_embeddings(self):
        return self.model.transformer.ff_out


class DreamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        for name in common.get_backend("dream").attention_suffixes:
            setattr(self.self_attn, name, nn.Linear(8, 8))
        self.mlp = nn.Module()
        for name in common.get_backend("dream").mlp_suffixes:
            setattr(self.mlp, name, nn.Linear(8, 8))


class TinyDream(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DreamBlock()])
        self.lm_head = nn.Linear(8, 16)

    def get_output_embeddings(self):
        return self.lm_head


class CoreTests(unittest.TestCase):
    def test_backend_target_enumeration_excludes_output_head(self):
        llada = list(common.iter_target_linears(TinyLLaDA(), "llada"))
        dream = list(common.iter_target_linears(TinyDream(), "dream"))
        self.assertEqual(len(llada), 7)
        self.assertEqual(len(dream), 7)
        self.assertFalse(any(name.endswith("lm_head") for name, _ in dream))
        self.assertFalse(any(name == "model.transformer.ff_out" for name, _ in llada))

    def test_random_t_is_deterministic_and_only_overwrites_with_mask(self):
        windows = torch.arange(48).reshape(3, 16)
        first = apply_noise(windows, seed=42, mask_id=999)
        second = apply_noise(windows, seed=42, mask_id=999)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1:], second[1:])
        valid = (first[0] == windows) | (first[0] == 999)
        self.assertTrue(bool(valid.all()))

    def test_whitening_factorizations_are_equivalent(self):
        torch.manual_seed(0)
        weight = torch.randn(24, 16, dtype=torch.float64)
        samples = torch.randn(256, 16, dtype=torch.float64)
        covariance = samples.T @ samples / samples.shape[0]
        a_chol, b_chol, _ = whiten_truncate(
            weight, covariance, 0.7, decomp="cholesky",
            solve_dtype=torch.float64, out_dtype=torch.float64,
        )
        a_eigh, b_eigh, _ = whiten_truncate(
            weight, covariance, 0.7, decomp="eigh",
            solve_dtype=torch.float64, out_dtype=torch.float64,
        )
        self.assertTrue(
            torch.allclose(a_chol @ b_chol, a_eigh @ b_eigh, atol=1e-9, rtol=1e-9)
        )

    def test_rank_and_memory_bins(self):
        self.assertEqual(common.rank_from_ratio(0.5, 12, 12), 3)
        modules = [("a", nn.Linear(4, 4)), ("b", nn.Linear(4, 4))]
        self.assertEqual(len(plan_bins(modules, budget_bytes=64)), 2)

    def test_evaluation_dry_commands_select_backend_harness(self):
        with tempfile.TemporaryDirectory() as output:
            llada, _, _ = build_command(
                "llada", "piqa", "ref", "model", None, 1, "2", output
            )
            dream, _, _ = build_command(
                "dream", "piqa", "ref", "model", None, 1, "2", output
            )
        self.assertTrue(any(value.endswith("llada_harness.py") for value in llada))
        self.assertTrue(any(value.endswith("dream_harness.py") for value in dream))


if __name__ == "__main__":
    unittest.main()
