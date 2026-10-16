import tempfile
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from trajmc import common
from trajmc.calibration import apply_noise
from trajmc.compression import (
    eigh_factor,
    plan_bins,
    relative_error,
    whiten_truncate,
)
from trajmc.sampling import (
    apply_forward_mask,
    dream_rollout_states,
    llada_rollout_states,
    resolve_scheme,
    rollout_call_indices,
    rollout_call_indices_for_transfers,
    uniform_grid,
)
from trajmc.analysis.calibration_ablation import audit_blob
from trajmc.analysis.covariance_estimation import collect_moments, compare_moments


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

    def test_grid_and_random_scheme_resolution(self):
        self.assertEqual(resolve_scheme("ours", None), ("random_t", "ours"))
        self.assertEqual(resolve_scheme(None, "grid_t"), ("grid_t", "grid_t"))
        grid = uniform_grid(8)
        self.assertTrue(torch.equal(
            rollout_call_indices(grid, 8), torch.arange(7, -1, -1)
        ))

    def test_relative_error_is_zero_for_an_exact_factorisation(self):
        torch.manual_seed(0)
        W = torch.randn(12, 8)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        A, B = U * S, Vt
        self.assertLess(relative_error(W, A, B), 1e-6)

    def test_rcond_floor_is_a_no_op_on_a_well_conditioned_covariance(self):
        # The paper claim the floor rests on: it only touches directions the
        # calibration never resolved.  On a spectrum far above the threshold it
        # must change nothing, so a shared floor cannot quietly alter the arms
        # whose statistics were fine.
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(16, 16))
        spectrum = torch.linspace(1.0, 4.0, 16, dtype=torch.float64)
        XtX = (Q.double() * spectrum) @ Q.double().T
        plain = eigh_factor(XtX, rcond=0.0)
        floored = eigh_factor(XtX, rcond=1e-6)
        self.assertTrue(torch.allclose(plain, floored, atol=1e-12))

    def test_rcond_floor_rescues_a_numerically_singular_covariance(self):
        # A miniature of the Dream failure.  A block of directions sits far
        # below float32 resolution -- as Dream's shared MLP-input covariance
        # does, at lambda_min/lambda_max = 3e-19 -- so their inverse square
        # roots are square roots of accumulation noise.  A single tiny
        # eigenvalue is not enough to reproduce it and would make this test
        # pass for the wrong reason.
        torch.manual_seed(0)
        dim = 64
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        spectrum = torch.full((dim,), 4.0, dtype=torch.float64)
        spectrum[: dim // 8] = 1e-18
        XtX = ((Q.double() * spectrum) @ Q.double().T).float()
        W = torch.randn(dim, dim)

        A0, B0, _ = whiten_truncate(W, XtX, 0.8, decomp="eigh", rcond=0.0)
        A1, B1, _ = whiten_truncate(W, XtX, 0.8, decomp="eigh", rcond=1e-8)
        Ai, Bi, _ = whiten_truncate(W, XtX, 0.8, decomp="identity")

        unfloored, floored = relative_error(W, A0, B0), relative_error(W, A1, B1)
        optimal = relative_error(W, Ai, Bi)
        self.assertGreater(unfloored, 1.0)
        self.assertLessEqual(floored, 1.0)
        # Plain SVD minimises this norm, so the floored solution cannot beat it;
        # landing just above it is what "the solve no longer diverges" means.
        self.assertGreaterEqual(floored, optimal - 1e-6)
        self.assertLess(floored, 1.5 * optimal)

    def test_rcond_floor_bounds_the_factor_conditioning(self):
        torch.manual_seed(0)
        dim = 12
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        spectrum = torch.full((dim,), 1.0, dtype=torch.float64)
        spectrum[0] = 1e-20
        XtX = (Q.double() * spectrum) @ Q.double().T
        rcond = 1e-8
        L = eigh_factor(XtX, rcond=rcond)
        values = torch.linalg.svdvals(L)
        self.assertLessEqual(
            float(values.max() / values.min()), 1.0 / math.sqrt(rcond) * 1.01
        )

    def test_fixed_t_resolves_and_collapses_the_timestep_distribution(self):
        self.assertEqual(resolve_scheme(None, "fixed_t"), ("fixed_t", "fixed_t"))
        windows = torch.arange(4096).reshape(2, 2048)
        timesteps = torch.full((2,), 0.5, dtype=torch.float64)
        noised = apply_forward_mask(windows, timesteps, mask_id=99, seed=7)
        rates = noised.eq(99).float().mean(dim=1)
        # Every row sits at the same nominal t, so the realised mask rates
        # concentrate there instead of spreading over [0, 1] as random_t does.
        self.assertTrue(bool(((rates - 0.5).abs() < 0.05).all()))

    def test_fixed_t_at_zero_reproduces_the_clean_arm_exactly(self):
        # The t = 0 cell of the fixed-t ablation is not run separately: it is
        # the existing clean_t0 arm.  That reuse is only sound while the two
        # constructions agree byte for byte.
        windows = torch.arange(64).reshape(4, 16)
        noised = apply_forward_mask(
            windows, torch.zeros(4, dtype=torch.float64), mask_id=99, seed=7
        )
        self.assertTrue(torch.equal(noised, windows))

    def test_prefix_grid_preserves_prefix_and_masks_suffix(self):
        windows = torch.arange(32).reshape(4, 8)
        noised = apply_forward_mask(
            windows, uniform_grid(4), mask_id=99, seed=7, prefix_ratio=0.25
        )
        self.assertTrue(torch.equal(noised[:, :2], windows[:, :2]))
        self.assertTrue(torch.equal(noised[-1, 2:], torch.full((6,), 99)))

    def test_rollout_mapping_tracks_nonuniform_transfer_schedule(self):
        transfers = [2] * 128 + [1] * 128
        calls = rollout_call_indices_for_transfers(uniform_grid(16), transfers)
        self.assertEqual(
            calls.tolist(),
            [232, 208, 184, 160, 136, 120, 108, 96,
             84, 72, 60, 48, 36, 24, 12, 0],
        )
        remaining = 384 - torch.tensor(
            [0, *torch.tensor(transfers).cumsum(0)[:-1].tolist()]
        )
        self.assertTrue(torch.equal(remaining[calls], torch.arange(24, 385, 24)))

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
