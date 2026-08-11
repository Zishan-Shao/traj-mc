import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from trajmc import common
from trajmc.calibration import apply_noise
from trajmc.compression import plan_bins, whiten_truncate
from trajmc.sampling import (
    apply_forward_mask,
    dream_rollout_states,
    llada_rollout_states,
    resolve_scheme,
    rollout_call_indices,
    uniform_grid,
)
from eval.run import build_command
from analysis.calibration_ablation import audit_blob
from analysis.covariance_estimation import collect_moments, compare_moments


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


class TinyRolloutModel(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(scale_logits=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.bias.zero_()
            self.lm_head.bias[2] = 10.0

    @property
    def device(self):
        return self.embedding.weight.device

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class TinyDreamRolloutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def device(self):
        return self.anchor.device

    def diffusion_generate(
        self, prompt, max_new_tokens, steps, mask_token_id,
        generation_tokens_hook_func, **_kwargs
    ):
        state = torch.full(
            (prompt.shape[0], prompt.shape[1] + max_new_tokens),
            mask_token_id,
            dtype=torch.long,
            device=prompt.device,
        )
        state[:, :prompt.shape[1]] = prompt
        state = generation_tokens_hook_func(None, state, None)
        for step in range(steps):
            masked = torch.nonzero(state[0].eq(mask_token_id), as_tuple=False)
            if masked.numel():
                state[0, masked[0, 0]] = 2
            state = generation_tokens_hook_func(step, state, None)
        return SimpleNamespace(sequences=state)


class TinyCovarianceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.proj = nn.Linear(6, 6)
        self.lm_head = nn.Linear(6, 16)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        hidden = self.proj(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


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

    def test_grid_and_random_scheme_resolution(self):
        self.assertEqual(resolve_scheme("ours", None), ("random_t", "ours"))
        self.assertEqual(resolve_scheme(None, "grid_t"), ("grid_t", "grid_t"))
        grid = uniform_grid(8)
        self.assertTrue(torch.equal(
            rollout_call_indices(grid, 8), torch.arange(7, -1, -1)
        ))

    def test_prefix_grid_preserves_prefix_and_masks_suffix(self):
        windows = torch.arange(32).reshape(4, 8)
        noised = apply_forward_mask(
            windows, uniform_grid(4), mask_id=99, seed=7, prefix_ratio=0.25
        )
        self.assertTrue(torch.equal(noised[:, :2], windows[:, :2]))
        self.assertTrue(torch.equal(noised[-1, 2:], torch.full((6,), 99)))

    def test_llada_rollout_states_contain_model_feedback(self):
        model = TinyRolloutModel()
        clean = (torch.arange(24).reshape(3, 8) % 10).long()
        states, metadata = llada_rollout_states(
            model,
            clean,
            uniform_grid(3),
            mask_id=15,
            prefix_ratio=0.25,
            steps=3,
        )
        self.assertEqual(metadata["rollout_call_indices"], [2, 1, 0])
        self.assertTrue(torch.equal(states[:, :2], clean[:, :2]))
        self.assertTrue(torch.equal(states[-1, 2:], torch.full((6,), 15)))
        revealed = states[:, 2:].ne(15)
        self.assertTrue(bool((revealed & states[:, 2:].ne(clean[:, 2:])).any()))

    def test_dream_rollout_uses_native_generation_hook(self):
        clean = (torch.arange(24).reshape(3, 8) % 10).long()
        states, metadata = dream_rollout_states(
            TinyDreamRolloutModel(),
            clean,
            uniform_grid(3),
            mask_id=15,
            prefix_ratio=0.25,
            steps=3,
        )
        self.assertEqual(metadata["rollout_call_indices"], [2, 1, 0])
        self.assertTrue(torch.equal(states[:, :2], clean[:, :2]))
        self.assertTrue(torch.equal(states[-1, 2:], torch.full((6,), 15)))
        self.assertEqual(int(states[0, 2:].eq(2).sum()), 2)

    def test_ablation_audit_and_covariance_identity(self):
        clean = torch.arange(32).reshape(4, 8)
        states = apply_forward_mask(
            clean, uniform_grid(4), mask_id=99, seed=3, prefix_ratio=0.25
        )
        blob = {
            "scheme": "grid_t_prefix",
            "windows_pre": clean,
            "input_ids": states,
            "mask_id": 99,
            "prefix_length": 2,
            "per_sample_t": uniform_grid(4).tolist(),
            "model_prediction_feedback": False,
        }
        row = audit_blob("grid", blob, clean)
        self.assertTrue(row["grid_exact"])
        reference = {"layer": torch.eye(4, dtype=torch.float64)}
        layers, aggregate = compare_moments(reference, reference)
        self.assertEqual(layers["layer"]["relative_frobenius_error"], 0.0)
        self.assertEqual(aggregate["mean_relative_trace_error"], 0.0)

    def test_projected_covariance_collector(self):
        model = TinyCovarianceModel()
        moments, counts = collect_moments(
            model,
            torch.arange(16).reshape(2, 8),
            [("proj", model.proj)],
            {"proj": torch.tensor([0, 2, 4])},
            device="cpu",
            batch_size=1,
        )
        self.assertEqual(moments["proj"].shape, (3, 3))
        self.assertEqual(counts["proj"], 16)
        self.assertTrue(torch.allclose(moments["proj"], moments["proj"].T))


if __name__ == "__main__":
    unittest.main()
