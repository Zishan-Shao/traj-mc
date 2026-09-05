"""CPU tests for the Traj-SVD diagnostics (RQ1 subspaces, RQ2 reconstruction)."""

import importlib.util
import math
import os
import unittest

import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from trajmc import common
from trajmc.calibration import artifact_prefix
from trajmc.compression import whiten_truncate
from analysis.gen_reconstruction import (
    measure,
    paired_bootstrap,
    ratio,
    stage_masks,
)
from trajmc.sampling import (
    LLADA_EOS_ID,
    LLADA_EOT_ID,
    _select_predictions,
    apply_forward_mask,
    assert_rollout_not_degenerate,
    block_transfer_schedule,
    llada_block_rollout_states,
    revealed_token_diagnostics,
    uniform_grid,
)
from trajmc.calibration import apply_noise
from analysis.subspace_distance import (
    block_index,
    kept_input_space,
    orthonormal_basis,
    select_layers,
    subspace_metrics,
    top_eigenspace,
)


HIDDEN = 8
VOCAB = 16


class RunnableBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in common.get_backend("llada").all_suffixes:
            setattr(self, name, nn.Linear(HIDDEN, HIDDEN, bias=False))

    def forward(self, hidden):
        for name in common.get_backend("llada").all_suffixes:
            hidden = getattr(self, name)(hidden)
        return hidden


class RunnableLLaDA(nn.Module):
    """Tiny model with the LLaDA target-Linear layout and a real forward."""

    def __init__(self, n_blocks=2):
        super().__init__()
        self.config = SimpleNamespace(scale_logits=False)
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.blocks = nn.ModuleList(
            [RunnableBlock() for _ in range(n_blocks)]
        )
        self.model.transformer.ff_out = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_output_embeddings(self):
        return self.model.transformer.ff_out

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        for block in self.model.transformer.blocks:
            hidden = block(hidden)
        return self.model.transformer.ff_out(hidden)


def exact_factors(model, layers):
    """Rank-preserving A/B so that A @ B reproduces W exactly."""
    factors = {}
    for name, module in layers:
        W = module.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        root = S.sqrt()
        factors[name] = ((U * root), (torch.diag(root) @ Vt))
    return factors


def truncated_factors(model, layers, ratio_value=0.5):
    factors = {}
    for name, module in layers:
        W = module.weight.data.float()
        A, B, _ = whiten_truncate(W, None, ratio_value, decomp="identity")
        factors[name] = (A, B)
    return factors


class GenReconstructionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = RunnableLLaDA().eval()
        self.backend = common.get_backend("llada")
        self.layers = list(common.iter_target_linears(self.model, self.backend))
        self.states = torch.randint(0, VOCAB, (5, 6))

    def test_exact_factorisation_has_zero_error(self):
        factors = exact_factors(self.model, self.layers)
        names, num, den = measure(
            self.model, self.states, self.layers, factors, "cpu", bins=1
        )
        self.assertEqual(len(names), len(self.layers))
        self.assertTrue((den > 0).all())
        self.assertLess(ratio(num, den), 1e-8)

    def test_matches_hand_computed_error(self):
        factors = truncated_factors(self.model, self.layers)
        names, num, den = measure(
            self.model, self.states, self.layers, factors, "cpu", bins=1
        )
        # Recompute layer 0 of block 0 by hand from its captured input.
        target_name = names[0]
        target = dict(self.layers)[target_name]
        captured = []
        handle = target.register_forward_hook(
            lambda _m, inputs, _o: captured.append(inputs[0].detach().clone())
        )
        with torch.no_grad():
            for row in range(self.states.shape[0]):
                self.model(self.states[row : row + 1])
        handle.remove()
        A, B = factors[target_name]
        W = target.weight.data.float()
        for row, hidden in enumerate(captured):
            flat = hidden.reshape(-1, HIDDEN).float()
            dense = flat @ W.T
            low_rank = flat @ B.T @ A.T
            self.assertAlmostEqual(
                num[row, 0], float((dense - low_rank).pow(2).sum()), places=3
            )
            self.assertAlmostEqual(den[row, 0], float(dense.pow(2).sum()), places=3)

    def test_bias_is_excluded_from_the_dense_reference(self):
        for _, module in self.layers:
            module.bias = nn.Parameter(torch.randn(HIDDEN))
        factors = exact_factors(self.model, self.layers)
        _, num, _ = measure(
            self.model, self.states, self.layers, factors, "cpu", bins=1
        )
        # A @ B reproduces W but carries no bias; the bias must be subtracted
        # from the dense output, otherwise every layer shows ||bias||^2 error.
        self.assertLess(float(num.max()), 1e-6)

    def test_binning_does_not_change_the_measurement(self):
        factors = truncated_factors(self.model, self.layers)
        names_one, num_one, den_one = measure(
            self.model, self.states, self.layers, factors, "cpu", bins=1
        )
        names_three, num_three, den_three = measure(
            self.model, self.states, self.layers, factors, "cpu", bins=3
        )
        self.assertEqual(names_one, names_three)
        np.testing.assert_allclose(num_one, num_three, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(den_one, den_three, rtol=1e-9, atol=1e-9)

    def test_partially_compressed_arm_measures_only_saved_layers(self):
        factors = truncated_factors(self.model, self.layers[:3])
        names, num, den = measure(
            self.model, self.states, self.layers, factors, "cpu", bins=1
        )
        self.assertEqual(names, [name for name, _ in self.layers[:3]])
        self.assertEqual(num.shape, (self.states.shape[0], 3))


class StatisticsTest(unittest.TestCase):
    def test_stage_masks_partition_every_row(self):
        ratios = np.linspace(0.0, 1.0, 30)
        stages = stage_masks(ratios)
        stacked = np.stack(list(stages.values()))
        np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(30, dtype=int))
        self.assertGreater(ratios[stages["early"]].mean(),
                           ratios[stages["late"]].mean())

    def test_paired_bootstrap_on_identical_arms_is_centred_on_zero(self):
        rng = np.random.default_rng(0)
        num = rng.random((40, 3))
        den = rng.random((40, 3)) + 1.0
        result = paired_bootstrap(num, den, num, den, draws=200, seed=1)
        self.assertEqual(result["delta_E_gen"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertFalse(result["improves"])

    def test_paired_bootstrap_detects_a_uniform_improvement(self):
        rng = np.random.default_rng(0)
        den = rng.random((60, 2)) + 1.0
        base = den * 0.4
        better = den * 0.1
        result = paired_bootstrap(base, den, better, den, draws=400, seed=2)
        self.assertAlmostEqual(result["delta_E_gen"], 0.3, places=6)
        self.assertTrue(result["improves"])
        self.assertGreater(result["ci95"][0], 0.0)


class SubspaceTest(unittest.TestCase):
    def test_identical_and_orthogonal_extremes(self):
        torch.manual_seed(0)
        basis = orthonormal_basis(torch.randn(20, 5))
        same = subspace_metrics(basis, basis)
        self.assertAlmostEqual(same["normalised_projector_frobenius"], 0.0, places=8)
        self.assertAlmostEqual(same["mean_principal_angle_deg"], 0.0, places=4)
        full = orthonormal_basis(torch.randn(20, 10))
        apart = subspace_metrics(full[:, :5], full[:, 5:])
        self.assertAlmostEqual(
            apart["normalised_projector_frobenius"], 1.0, places=8
        )
        self.assertAlmostEqual(apart["mean_principal_angle_deg"], 90.0, places=4)

    def test_isotropic_rescaling_leaves_the_kept_subspace_fixed(self):
        torch.manual_seed(0)
        W = torch.randn(16, 24)
        X = torch.randn(400, 24)
        sigma = (X.T @ X) / 400
        kept = kept_input_space(W, sigma, 0.5, "cholesky")
        rescaled = kept_input_space(W, 7.3 * sigma, 0.5, "cholesky")
        distance = subspace_metrics(kept, rescaled)[
            "normalised_projector_frobenius"
        ]
        self.assertLess(distance, 1e-4)

    def test_a_different_covariance_moves_the_kept_subspace(self):
        torch.manual_seed(0)
        W = torch.randn(16, 24)
        X = torch.randn(400, 24)
        Y = torch.randn(400, 24) @ torch.diag(torch.linspace(0.1, 5.0, 24))
        kept = kept_input_space(W, (X.T @ X) / 400, 0.5, "cholesky")
        other = kept_input_space(W, (Y.T @ Y) / 400, 0.5, "cholesky")
        self.assertGreater(
            subspace_metrics(kept, other)["normalised_projector_frobenius"], 0.1
        )

    def test_top_eigenspace_returns_the_leading_directions(self):
        values = torch.tensor([5.0, 3.0, 1.0])
        vectors = torch.linalg.qr(torch.randn(3, 3))[0]
        sigma = vectors @ torch.diag(values) @ vectors.T
        top = top_eigenspace(sigma, 2)
        self.assertEqual(tuple(top.shape), (3, 2))
        distance = subspace_metrics(
            top, vectors[:, :2].to(torch.float64)
        )["normalised_projector_frobenius"]
        self.assertLess(distance, 1e-6)

    def test_layer_selection_and_depth_index(self):
        model = RunnableLLaDA(n_blocks=4)
        backend = common.get_backend("llada")
        chosen = select_layers(model, backend, ["q_proj"], 2, [])
        self.assertEqual(len(chosen), 2)
        self.assertTrue(all(name.endswith("q_proj") for name, _ in chosen))
        indices = [block_index(name, backend.block_marker) for name, _ in chosen]
        self.assertEqual(indices, [0, 2])


def _released_mcs():
    """Import the vendored Quant-dLLM MCS, or skip if it is not checked out."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "baselines", "quant_dllm", "utils", "mcs.py",
    )
    if not os.path.exists(path):
        raise unittest.SkipTest("baselines/quant_dllm is not vendored here")
    spec = importlib.util.spec_from_file_location("released_mcs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SchemeIdentityTest(unittest.TestCase):
    """Which repo scheme is which published method.

    The five-arm table maps `mcs` to `grid_t_prefix` and `traj` to `random_t`.
    That mapping is only defensible if `grid_t_prefix` really is the released
    baseline, so pin the equality against the vendored source rather than
    against a description of it.
    """

    MASK = 126336

    def setUp(self):
        torch.manual_seed(0)
        self.windows = torch.randint(0, 50000, (32, 256))

    def test_grid_t_prefix_reproduces_released_mcs(self):
        mcs = _released_mcs()
        n = self.windows.shape[0]
        ours = apply_forward_mask(
            self.windows, uniform_grid(n), self.MASK, 42, prefix_ratio=0.25
        )
        theirs = torch.stack([
            mcs.apply_mcs(
                self.windows[i : i + 1], self.MASK, i, n,
                prefix_ratio=0.25, seed=42,
            )[0][0]
            for i in range(n)
        ])
        self.assertTrue(torch.equal(ours, theirs))

    def test_uniform_grid_matches_the_released_timestep_rule(self):
        mcs = _released_mcs()
        n = self.windows.shape[0]
        grid = uniform_grid(n)
        for index in range(n):
            self.assertAlmostEqual(
                float(grid[index]), mcs.timestep_for_sample(index, n), places=12
            )

    def test_grid_t_without_a_prefix_is_not_the_released_baseline(self):
        mcs = _released_mcs()
        n = self.windows.shape[0]
        no_prefix = apply_forward_mask(
            self.windows, uniform_grid(n), self.MASK, 42, prefix_ratio=0.0
        )
        theirs = torch.stack([
            mcs.apply_mcs(
                self.windows[i : i + 1], self.MASK, i, n,
                prefix_ratio=0.25, seed=42,
            )[0][0]
            for i in range(n)
        ])
        self.assertFalse(torch.equal(no_prefix, theirs))
        prefix_length = int(0.25 * self.windows.shape[1])
        # The published baseline never masks its prefix; the 0% arm always can.
        self.assertEqual(
            int(theirs[:, :prefix_length].eq(self.MASK).any(1).sum()), 0
        )
        self.assertGreater(
            int(no_prefix[:, :prefix_length].eq(self.MASK).any(1).sum()), 0
        )

    def test_traj_arm_masks_the_whole_window(self):
        # Traj-SVD is random_t: iid t, no protected region anywhere.
        states, timesteps, _, _ = apply_noise(self.windows, 42, self.MASK)
        prefix_length = int(0.25 * self.windows.shape[1])
        self.assertGreater(
            int(states[:, :prefix_length].eq(self.MASK).any(1).sum()), 0
        )
        self.assertEqual(len(timesteps), self.windows.shape[0])
        self.assertTrue(all(0.0 <= t <= 1.0 for t in timesteps))


class BlockRolloutTest(unittest.TestCase):
    """The deployed sampler reveals only inside the current block.

    Treating the whole generation region as one block is what collapsed
    LLaDA-8B-Instruct into EOS padding; these tests pin the block restriction
    that the official evaluation guide relies on.
    """

    MASK = VOCAB - 1

    def setUp(self):
        torch.manual_seed(0)
        self.model = RunnableLLaDA().eval()
        self.prompts = torch.randint(0, VOCAB - 1, (4, 6))

    def test_schedule_matches_the_official_shapes(self):
        for gen, block, steps, blocks, per_block in (
            (256, 8, 256, 32, 8),      # official GSM8K block diffusion
            (512, 64, 512, 8, 64),     # official MATH block diffusion
        ):
            transfers, num_blocks, steps_per_block = block_transfer_schedule(
                gen, block, steps
            )
            self.assertEqual((num_blocks, steps_per_block), (blocks, per_block))
            self.assertEqual(len(transfers), steps)
            self.assertEqual(sum(transfers), gen)

    def test_indivisible_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            block_transfer_schedule(256, 7, 256)
        with self.assertRaises(ValueError):
            block_transfer_schedule(256, 8, 100)

    def test_nothing_is_revealed_beyond_the_current_block(self):
        gen, block, steps = 8, 2, 8
        timesteps = torch.tensor([0.9, 0.6, 0.4, 0.1])
        states, meta = llada_block_rollout_states(
            self.model, self.prompts, timesteps, self.MASK, gen, block, steps
        )
        prompt_length = self.prompts.shape[1]
        self.assertEqual(states.shape, (4, prompt_length + gen))
        self.assertEqual(meta["rollout_block_length"], block)
        for row, call in enumerate(meta["rollout_call_indices"]):
            current_block = call // meta["rollout_steps_per_block"]
            frontier = prompt_length + (current_block + 1) * block
            beyond = states[row, frontier:]
            self.assertTrue(
                bool(beyond.eq(self.MASK).all()),
                f"row {row} revealed a token past block {current_block}",
            )

    def test_the_prompt_is_never_modified(self):
        states, _ = llada_block_rollout_states(
            self.model, self.prompts, torch.tensor([0.9, 0.6, 0.4, 0.1]),
            self.MASK, 8, 2, 8,
        )
        prompt_length = self.prompts.shape[1]
        self.assertTrue(torch.equal(states[:, :prompt_length], self.prompts))

    def test_a_masked_prompt_is_refused(self):
        poisoned = self.prompts.clone()
        poisoned[0, 0] = self.MASK
        with self.assertRaises(ValueError):
            llada_block_rollout_states(
                self.model, poisoned, torch.tensor([0.9, 0.6, 0.4, 0.1]),
                self.MASK, 8, 2, 8,
            )

    def test_t_near_one_returns_the_untouched_initial_state(self):
        states, _ = llada_block_rollout_states(
            self.model, self.prompts, torch.ones(4), self.MASK, 8, 2, 8
        )
        prompt_length = self.prompts.shape[1]
        self.assertTrue(bool(states[:, prompt_length:].eq(self.MASK).all()))


class EosSuppressionTest(unittest.TestCase):
    """generate.py's two EOS switches, whose order is easy to get backwards.

    ``logits_eos_inf`` is applied before the argmax, so EOS can never be
    predicted.  ``confidence_eos_eot_inf`` is applied after it, so EOS may
    still be predicted but its confidence collapses and top-k never picks it.
    MATH runs with the second one on; GSM8K runs with neither.
    """

    def logits(self, rows=5):
        torch.manual_seed(0)
        values = torch.randn(rows, 126464)
        values[:, LLADA_EOS_ID] = 50.0
        values[:, LLADA_EOT_ID] = 49.0
        return values

    def test_no_suppression_predicts_and_trusts_eos(self):
        predictions, confidence = _select_predictions(self.logits(), False, False)
        self.assertTrue(bool((predictions == LLADA_EOS_ID).all()))
        self.assertGreater(float(confidence.mean()), 0.5)

    def test_confidence_switch_keeps_the_prediction_but_kills_the_score(self):
        predictions, confidence = _select_predictions(self.logits(), False, True)
        self.assertTrue(bool((predictions == LLADA_EOS_ID).all()))
        self.assertEqual(float(confidence.max()), 0.0)

    def test_logits_switch_stops_eos_being_predicted_at_all(self):
        predictions, _ = _select_predictions(self.logits(), True, False)
        self.assertEqual(int((predictions == LLADA_EOS_ID).sum()), 0)
        # logits_eos_inf touches EOS only; EoT is still reachable.
        self.assertTrue(bool((predictions == LLADA_EOT_ID).all()))

    def test_math_and_gsm8k_carry_the_harness_switches(self):
        from trajmc.prompts import official_sampler

        gsm = official_sampler("gsm8k_test")
        self.assertFalse(gsm["logits_eos_inf"])
        self.assertFalse(gsm["confidence_eos_eot_inf"])
        math = official_sampler("math500")
        self.assertFalse(math["logits_eos_inf"])
        self.assertTrue(math["confidence_eos_eot_inf"])


class RolloutDegeneracyTest(unittest.TestCase):
    """A rollout that floods the suffix with one token must not be recorded.

    Observed for real: LLaDA-Instruct given a raw C4 prefix and a single
    1536-token block filled 99.93% of revealed positions with <|endoftext|>.
    The existing manifest field could not surface it -- "wrote other text" and
    "wrote nothing" both push prediction_mismatch_fraction to ~1.0.
    """

    MASK, EOS, PREFIX = 126336, 126081, 4

    def states(self, suffix_rows):
        prefix = torch.full((len(suffix_rows), self.PREFIX), 7, dtype=torch.long)
        return torch.cat([prefix, torch.tensor(suffix_rows)], dim=1)

    def test_flooded_rollout_is_rejected(self):
        flooded = self.states([[3, self.EOS, self.EOS, self.EOS, self.EOS]] * 8)
        diagnostics = revealed_token_diagnostics(flooded, self.MASK, self.PREFIX)
        self.assertGreater(diagnostics["top_revealed_token_share"], 0.5)
        self.assertEqual(diagnostics["median_distinct_tokens_per_row"], 2)
        with self.assertRaises(RuntimeError) as caught:
            assert_rollout_not_degenerate(diagnostics)
        self.assertIn("degenerate rollout", str(caught.exception))
        self.assertIn(str(self.EOS), str(caught.exception))

    def test_varied_generation_passes(self):
        torch.manual_seed(0)
        varied = torch.randint(100, 900, (8, 64))
        states = torch.cat(
            [torch.full((8, self.PREFIX), 7, dtype=torch.long), varied], dim=1
        )
        diagnostics = revealed_token_diagnostics(states, self.MASK, self.PREFIX)
        self.assertLess(diagnostics["top_revealed_token_share"], 0.5)
        assert_rollout_not_degenerate(diagnostics)

    def test_masked_positions_are_not_counted_as_revealed(self):
        rows = torch.full((4, 8), self.MASK, dtype=torch.long)
        rows[:, 0] = 11
        states = torch.cat(
            [torch.full((4, self.PREFIX), 7, dtype=torch.long), rows], dim=1
        )
        diagnostics = revealed_token_diagnostics(states, self.MASK, self.PREFIX)
        self.assertEqual(diagnostics["revealed_tokens"], 4)
        self.assertEqual(diagnostics["unique_revealed_tokens"], 1)

    def test_prefix_is_excluded_from_the_diagnosis(self):
        # The visible prefix is clean text by construction; counting it would
        # dilute the very flooding this guard exists to catch.
        flooded = self.states([[self.EOS] * 5] * 8)
        diagnostics = revealed_token_diagnostics(flooded, self.MASK, self.PREFIX)
        self.assertEqual(diagnostics["revealed_tokens"], 40)
        self.assertEqual(diagnostics["top_revealed_token_share"], 1.0)

    def test_an_empty_rollout_is_reported_not_crashed(self):
        empty = torch.full((3, self.PREFIX + 5), self.MASK, dtype=torch.long)
        diagnostics = revealed_token_diagnostics(empty, self.MASK, self.PREFIX)
        self.assertEqual(diagnostics["revealed_tokens"], 0)
        self.assertIsNone(diagnostics["top_revealed_token_share"])
        assert_rollout_not_degenerate(diagnostics)


INSTRUCT_CHECKPOINT = "/zpool-00/home/tl356/LLaDA-8B-Instruct"


class TaskPromptTest(unittest.TestCase):
    """Prompts must match utils/eval_benchmarks.py, not merely resemble it.

    The prompt decides what the dense model writes, so an Actual-State rollout
    built on a different prompt records states the harness never visits. A
    5-shot random draw in `Question:/Answer:` form was used once; every number
    measured on it had to be discarded.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(INSTRUCT_CHECKPOINT):
            raise unittest.SkipTest("LLaDA-8B-Instruct checkpoint not present")
        from transformers import AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained(
            INSTRUCT_CHECKPOINT, trust_remote_code=True
        )

    def build(self, source, count=8, seed=42, **kwargs):
        from trajmc.prompts import build_task_prompts

        return build_task_prompts(self.tok, source, count, None, seed=seed, **kwargs)

    def test_prompt_text_is_the_harness_builder_verbatim(self):
        from trajmc.prompts import GSM8K_FEWSHOT, build_gsm8k_prompt

        question = "If Ann has 3 apples and buys 4 more, how many does she have?"
        expected = "\n\n".join(
            [f"Question: {q}\nLet's think step by step\nAnswer: {a}"
             for q, a in GSM8K_FEWSHOT]
            + [f"Question: {question}\nLet's think step by step\nAnswer:"]
        )
        self.assertEqual(build_gsm8k_prompt(question), expected)

    def test_gsm8k_uses_the_fixed_four_shot_block(self):
        from trajmc.prompts import GSM8K_FEWSHOT

        self.assertEqual(len(GSM8K_FEWSHOT), 4)
        _, meta = self.build("gsm8k_train")
        self.assertEqual(meta["prompt_num_fewshot"], 4)
        self.assertTrue(meta["prompt_fewshot_fixed"])
        self.assertIsNone(meta["prompt_fewshot_indices"])

    def test_a_different_shot_count_is_refused_for_gsm8k(self):
        with self.assertRaises(ValueError):
            self.build("gsm8k_train", num_fewshot=5)

    def test_official_sampler_settings_are_carried(self):
        from trajmc.prompts import official_sampler

        for source in ("gsm8k_train", "gsm8k_test"):
            self.assertEqual(
                official_sampler(source),
                {"gen_length": 256, "block_length": 8, "steps": 256,
                 "logits_eos_inf": False, "confidence_eos_eot_inf": False,
                 "num_fewshot": 4},
            )
        self.assertTrue(official_sampler("math500")["confidence_eos_eot_inf"])

    def test_each_source_alone_does_not_share_a_geometry(self):
        # Left as an explicit warning: the per-source minimum depends on which
        # questions were drawn, so the two arms would differ.
        _, train = self.build("gsm8k_train", seed=42)
        _, test = self.build("gsm8k_test", seed=1337)
        self.assertNotEqual(train["prompt_length"], test["prompt_length"])

    def test_shared_prompt_length_makes_the_arms_comparable(self):
        from trajmc.prompts import build_task_prompts, shared_prompt_length

        specs = [("gsm8k_train", 8, 42), ("gsm8k_test", 8, 1337)]
        length = shared_prompt_length(self.tok, specs)
        built = [
            build_task_prompts(self.tok, source, count, length, seed)[1]
            for source, count, seed in specs
        ]
        self.assertEqual(built[0]["prompt_length"], built[1]["prompt_length"])
        self.assertEqual(built[0]["prompt_length"], length)

    def test_rows_are_exactly_the_reported_length(self):
        ids, meta = self.build("gsm8k_train")
        self.assertEqual(ids.shape, (8, meta["prompt_length"]))
        self.assertLessEqual(meta["prompt_natural_length_max"], 1900)

    def test_no_pad_or_eos_token_is_ever_inserted(self):
        ids, _ = self.build("gsm8k_train")
        body = ids[:, 6:-7]  # strip the chat-template head and tail
        for token in ("<|endoftext|>", "<|eot_id|>"):
            self.assertEqual(
                int(body.eq(self.tok.convert_tokens_to_ids(token)).sum()), 0
            )

    def test_the_row_ends_with_the_generation_cue(self):
        ids, _ = self.build("gsm8k_train")
        self.assertIn("assistant", self.tok.decode(ids[0, -12:].tolist()))

    def test_math_shots_stay_disjoint_from_its_targets(self):
        _, meta = self.build("math500")
        self.assertEqual(meta["prompt_num_fewshot"], 4)
        self.assertFalse(
            set(meta["prompt_target_indices"]) & set(meta["prompt_fewshot_indices"])
        )


class ArtifactNameTest(unittest.TestCase):
    """The shell scripts ask this function for paths; pin its format.

    A shell-side re-derivation of these names used to miss the scheme and
    split-seed suffixes, and the mismatch only surfaced after the rollout it
    named had already run.
    """

    def name(self, scheme, **overrides):
        kwargs = dict(
            backend_name="llada_instruct", label=scheme, scheme=scheme,
            corpus="c4", nsamples=256, seed=42, sampling_seed=42,
            dry_run=False, prefix_ratio=0.25, rollout_steps=256,
            git_hash="abc1234",
        )
        kwargs.update(overrides)
        return artifact_prefix(**kwargs)

    def test_plain_schemes_carry_no_scheme_suffix(self):
        for scheme in ("clean_t0", "random_t", "grid_t"):
            self.assertEqual(
                self.name(scheme),
                f"llada_instruct_{scheme}_c4_n256_s42_full_abc1234",
            )

    def test_prefix_and_rollout_schemes_carry_their_suffixes(self):
        self.assertEqual(
            self.name("grid_t_prefix"),
            "llada_instruct_grid_t_prefix_p0p25_c4_n256_s42_full_abc1234",
        )
        self.assertEqual(
            self.name("rollout"),
            "llada_instruct_rollout_p0p25_steps256_c4_n256_s42_full_abc1234",
        )

    def test_split_sampling_seed_switches_the_seed_tag(self):
        self.assertIn("_ws42_ms7_", self.name("grid_t", sampling_seed=7))
        self.assertIn("_s42_", self.name("grid_t", sampling_seed=42))

    def test_grid_names_never_prefix_match_each_other(self):
        # A glob built from "grid_t" must not be able to pick up grid_t_prefix.
        self.assertFalse(
            self.name("grid_t_prefix").startswith(
                self.name("grid_t").split("_c4_")[0] + "_c4_"
            )
        )

    def test_dry_run_and_step_count_reach_the_name(self):
        self.assertIn("_dry_", self.name("clean_t0", dry_run=True))
        self.assertIn("_steps64_", self.name("rollout", rollout_steps=64))
        self.assertIn("_p0p5_", self.name("rollout", prefix_ratio=0.5))


class InstructBackendTest(unittest.TestCase):
    def test_instruct_backends_mirror_their_base_sibling(self):
        for instruct, base in (("llada_instruct", "llada"),
                               ("dream_instruct", "dream")):
            spec = common.get_backend(instruct)
            reference = common.get_backend(base)
            self.assertEqual(common.arch_family(spec), base)
            self.assertEqual(spec.mask_id, reference.mask_id)
            self.assertEqual(spec.all_suffixes, reference.all_suffixes)
            self.assertEqual(spec.expected_linears, reference.expected_linears)
            self.assertNotEqual(spec.model_id, reference.model_id)

    def test_weight_only_svd_equals_whitening_with_identity(self):
        torch.manual_seed(0)
        W = torch.randn(32, 48)
        A, B, k = whiten_truncate(W, None, 0.5, decomp="identity")
        A2, B2, k2 = whiten_truncate(W, torch.eye(48), 0.5, decomp="cholesky")
        self.assertEqual(k, k2)
        self.assertEqual(k, common.rank_from_ratio(0.5, 32, 48))
        self.assertTrue(torch.allclose(A @ B, A2 @ B2, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
