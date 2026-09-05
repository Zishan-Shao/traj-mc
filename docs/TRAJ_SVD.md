# Traj-SVD: does deployment-aligned calibration change the kept subspace?

Activation-aware low-rank compression minimises

```text
R_P(W~) = E_{h~P} ||(W - W~) h||^2 = ||(W - W~) L||_F^2,   Sigma_P = E[h h^T] = L L^T,
```

whose rank-`k` optimum is `W~* = (W L)_k L^{-1}`. The solver is fixed; the only
free choice is the activation distribution `P` that defines "optimal". A dLLM
never sees clean text at inference — it sees a sequence of partially masked
generation states. This study asks whether that changes the answer, and whether
the change survives to mathematical reasoning.

## Arms

Every arm shares the compressed layer scope, rank allocation, calibration
budget, decomposition, and clean pre-noise windows. Only the states used to
estimate `Sigma` differ.

| Arm | Scheme | States used for `Sigma` |
| --- | --- | --- |
| `weight_svd` | `--decomp identity` | none; plain truncated SVD of `W` |
| `clean` | `clean_t0` | clean calibration text, `t = 0` |
| `traj` | `random_t` | **Traj-SVD**: one `t ~ U(0,1)` per window, every token independently masked with probability `t` |
| `mcs` | `grid_t_prefix` | released Quant-dLLM MCS: stratified grid `1/N..1`, 25% visible prefix |
| `actual` | `rollout` | control: real dense reverse-sampler pre-forward states |

`actual` is an experimental **control**, not a deployable method: it requires
dense generation before compression. It bounds how much of the gap a synthetic
construction could close.

`mcs` is byte-identical to the released implementation. `grid_t_prefix` with
`prefix_ratio=0.25` reproduces `baselines/quant_dllm/utils/mcs.py::apply_mcs`
exactly -- same uniform grid `(i+1)/N`, same per-sample generator seeded
`seed + sample_index`, same protected prefix -- and the test suite pins that
equality. The released MCS keeps a visible prefix *by default*; an arm without
one is not the published baseline.

`grid_t` (stratified grid, 0% prefix) exists in the pipeline but has no row in
this table: it corresponds to no published method. It is the arm that would
separate the two axes below, and it is worth one run only if that attribution
is challenged.

### What separates `traj` from `mcs`

Two axes change at once between the method and its baseline.

| Axis | `traj` (`random_t`) | `mcs` (`grid_t_prefix`) |
| --- | --- | --- |
| Timestep design | iid `t ~ U(0,1)` | stratified grid `1/N..1` |
| Visible prefix | 0% -- the whole window is maskable | 25% -- first quarter never masked |

The two axes are not the same kind of difference:

- **Timestep design is one target distribution under two estimators.** Both
  integrate over `t ~ U(0,1)`; iid is unbiased Monte Carlo, the grid is
  stratified quadrature. Measured over 256 windows of length 2048 they agree on
  the marginal:

  | Scheme | mean full-window MASK rate | std |
  | --- | ---: | ---: |
  | `random_t`, iid | 0.473 | 0.289 |
  | `grid_t_prefix`, stratified grid | 0.502 | 0.289 |

  Identical spread, means apart by 0.03. The grid may have lower variance at a
  fixed budget; nothing here should claim stratification always wins, or always
  loses.
- **The visible prefix is a change of distribution.** 256 of 256 `random_t`
  windows carry MASK inside the first 512 positions; 0 of 256 MCS windows do.
  The full-window MASK rate drops from 0.502 to 0.376 purely because the prefix
  is protected, while the *suffix* rate is unchanged at 0.502.

**What this means for the writeup.** The two schemes sample the same timestep
marginal to within noise, so whatever separates `traj` from `mcs` comes mostly
from the absence of prefix protection, not from iid sampling being unbiased.
Do not sell iid unbiasedness as the headline mechanism: the measurement above
does not support it. Because both axes move together, a `traj` win is not by
itself attributable to either; state the comparison as
method-vs-published-baseline, which is what it is, and add `grid_t` only if a
reviewer asks which axis carried the effect.

## Gates

Stage A runs one model at one compression level and answers two questions
before any benchmark is launched.

**RQ1 — is the kept subspace different?** `analysis.subspace_distance`
recomputes, per Linear, the rank-`k` input subspace each calibration keeps and
measures its distance to the `rollout` subspace. Two subspaces are reported:
`kept_input` (row space of `B` in `W ~= A B`, what the deployed layer actually
keeps) and `sigma_eig` (top-`k` eigenspace of `Sigma`). Distances are projector
Frobenius norms normalised to `[0, 1]`, computed as
`sqrt(k1 + k2 - 2||Q1^T Q2||_F^2)` without materialising either projector, and
reported alongside `sigma_relative_trace` so that a pure activation-scale shift
is not mistaken for a subspace shift — the kept subspace is invariant to
isotropic rescaling of `Sigma`, and the test suite pins that.

The gate is

```text
d_sub(traj, actual) < d_sub(clean, actual).
```

**RQ2 — does it reconstruct real generation states better?**
`analysis.gen_reconstruction` replays held-out real rollout states through the
dense model and measures, per compressed Linear,

```text
E_gen = sum ||(W - A B) h||^2 / sum ||W h||^2.
```

The dense forward supplies `W h` as the layer's own output, so each arm costs
one extra low-rank matmul per layer rather than a second dense pass.

The gate is `E_gen(traj) < E_gen(clean)`, with

```text
GapClosed = (E_clean - E_traj) / (E_clean - E_actual).
```

If neither gate holds, the mechanism the paper claims does not exist and the
downstream matrix should not be run.

### Held-out prompts and the bootstrap unit

A rollout artifact stores exactly one state per clean window, so its rows are
independent: one prompt, one sampler state. That makes the row the correct
bootstrap unit, and it rules out treating several steps of one trajectory as
independent samples. Deltas against the `clean` arm use a paired bootstrap over
the shared row index.

Evaluation states must come from prompts no arm calibrated on.
`run_stage_a.sh` builds them with a different window seed, and
`analysis.gen_reconstruction --calib_manifest` hard-fails on any pre-noise
window hash shared with a calibration artifact.

## Running Stage A

```bash
STAGE=all RATIO=0.8 scripts/run_stage_a.sh llada_instruct
```

Defaults: `NSAMPLES=256`, `SEED=42` (calibration windows), `EVAL_SEED=1337`
(held-out states), `PREFIX_RATIO=0.25` (the MCS arm's prefix; the Traj arm has
none), `ROLLOUT_STEPS=256`. The two working points are `RATIO=0.8` and
`RATIO=0.6` -- retention fractions, i.e. 20% and 40% parameter reduction.
Stage A runs `0.8` only. `STAGE` may be
`calibrate`, `compress`, `measure`, or `all`; the later stages relocate the
artifacts the earlier ones wrote. Outputs land under
`results/stage_a/<backend>_r<ratio>/`.

Cost notes:

- the `rollout` arm and the held-out states each need dense generation, which
  dominates Stage A wall clock;
- `subspace_distance` holds one full `XtX` per artifact per selected layer on
  CPU. It defaults to one suffix swept over depth; use `--suffix` and
  `--block_stride` to control the footprint it prints on startup.
- `gen_reconstruction` holds one arm's `A`/`B` factors on the GPU beside the
  dense model. Raise `RECON_BINS` to split the layers over several passes if
  that does not fit.

## What Stage A cannot answer

Both gates are statements about *local* layerwise reconstruction. Whether
better local optimality preserves mathematical reasoning is a separate
question, measured downstream on GSM8K and MATH-500, and a stable local
improvement with mixed reasoning outcomes is a reportable result rather than a
failure.

## Open item: the method description in the plan contradicts the arm

The plan's section 1.3 describes Traj-SVD as constructing states from the real
sampler's generation schedule -- how much MASK remains at each step, where the
generation region is, how MASK positions are drawn. Two of those three hold for
`grid_t_prefix` and none of them hold for `random_t`, so as drafted the method
paragraph describes the MCS baseline rather than the method.

Execution follows section 0, which is explicit and operational: the study is
the `t = 0` versus `random_t` comparison at retention 0.8 and 0.6. Section 1.3
is therefore **marked for rewrite**, in this direction:

> Traj-SVD draws unbiased iid Monte Carlo samples from the forward diffusion
> marginal the model was trained on -- one `t ~ U(0,1)` per window, every token
> independently masked with probability `t` -- rather than evaluating a
> stratified quadrature grid over a sequence whose prefix has been frozen
> visible.

Shipping the current wording would let a reviewer line the method description
up against the baseline description and find them swapped.

## The rollout sampler configuration

The first Stage A submission (job 167372, LLaDA-Instruct, `RATIO=0.8`) produced
a **degenerate** Actual-State control and was cancelled. Of 195,840 tokens the
rollout revealed, 99.93% were `<|endoftext|>`; only 74 distinct tokens appeared
in total, and the median rollout revealed exactly one. The model wrote nothing.

That run fed a raw C4 prefix, with no chat template, to an *Instruct*
checkpoint, and asked for 1536 tokens as a single block. LLaDA's own evaluation
guide documents the failure mode it hit: LLaDA-8B-Instruct "generate[s]
excessive |EOS| tokens ... caused by the extensive |EOS| padding in the SFT
data", which block diffusion exists to suppress (GSM8K 69.4 -> 78.6).

### What the control actually showed

A 16-prompt control (job 168456) separated the two suspects at
`gen_length=256`, holding the task prompts fixed and varying only the block
length:

| Configuration | top token | EOS | unique ratio | median distinct/row | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| C4 prefix, 1536 single block (job 167372) | 99.93% | 99.94% | 0.00038 | 1 | collapsed |
| task prompt, `block_length=8` | 7.45% | 1.30% | 0.168 | 53 | healthy |
| task prompt, `block_length=256` | 7.29% | 1.15% | 0.165 | 53 | healthy |

**Block length is not what caused the collapse.** With deployment-format
prompts, a single 256-token block generates ordinary step-by-step reasoning.
The prompt format is the implicated cause; `gen_length` 1536 versus 256 is not
separated by this control, since job 167372 changed both at once.

Block length still matters, but for *quality*, not for collapse. At equal
reveal counts the single-block arm degrades visibly -- "First, calculate the
from the necklaces ... $ear\n#### 5" against the block-diffusion arm's
well-formed enumeration -- which is the same effect the official 69.4 -> 78.6
gap measures. The Actual-State control therefore uses the official deployed
configuration (`gen_length=256, block_length=8` for GSM8K;
`gen_length=512, block_length=64` for MATH) because fidelity to deployment is
its whole purpose, not because a longer block would collapse.

Note the official values: `block_length` is **8** for GSM8K and **64** for
MATH; 256 and 512 are the *generation* lengths. The pure-diffusion settings in
the same table (`block_length = gen_length`) reach 69.4 and 31.9 only with
`confidence_eos_eot_inf=True`.

### The guard

`revealed_token_diagnostics` and `assert_rollout_not_degenerate` run on every
rollout before its tensor is written, recording the statistics in the manifest
and refusing to save a flooded artifact unless `--allow_degenerate_rollout` is
passed. Five decoded samples are printed unconditionally: thresholds catch the
failure modes already known, reading the text catches the rest.

Limits sit an order of magnitude from both reference points -- genuine C4 text
scores top token 3.85%, unique ratio 0.108, median 350 per row. The guard is
deliberately a *collapse* detector, not a quality judge: it correctly stays
silent on the healthy-but-worse single-block arm above.

`prediction_mismatch_fraction` must never be used for this on its own. It sits
near 1.0 both when the sampler writes different text and when it writes
nothing, which is why the original failure survived unnoticed. It is recorded
as `None` for task-prompt rollouts, where there is no ground-truth
continuation, and carries a caveat field in every manifest.

The same parameters are what `run_sampling_ablation.sh` has always used, so any
earlier four-way ablation's `rollout` arm should be re-checked with
`analysis.void_rollout` before its numbers are used.

### Deployment states

`Actual-State` calibration draws GSM8K *train* questions; the held-out
evaluation states draw GSM8K *test*. Both share one fixed few-shot context --
deployment holds the prefix fixed and varies only the question -- so the
control and the states it is judged on have identical prompt geometry. The
context seed is independent of the target seed for exactly this reason.

The control is not part of any single-variable claim, so it does not share a
corpus with `clean`, `traj`, and `mcs`; window identity stays binding on those
three. The cost is that GapClosed's denominator spans both state source and
task domain, and the paper must say so: "Traj closes X% of the total clean-to-
deployment gap", not "X% of the state-source dimension". A C4-rollout secondary
oracle could separate the two later; it is not run now.

## Benchmark scope: main table vs mechanism diagnostic

Two groups of columns, and the distinction is load-bearing: only the first
group carries a claim about the method, and only it is protocol-frozen.

**Main table (spec-frozen).** Gen.Recon., GSM8K, MATH-500, MMLU, and SVAMP.
SVAMP is a pre-registered addition, not a column chosen after seeing results.

**Mechanism diagnostic (figure/appendix, supporting RQ3).** Split by how much
iterative generation the protocol actually performs, which is the axis the
diagnostic exists to isolate:

| End | Columns | Sampler |
| --- | --- | --- |
| long generation | GSM8K, SVAMP, MATH-500, ARC-C | 256-512 denoising steps |
| short generation | MMLU, HellaSwag, PIQA | 3 tokens, answer read out as an option letter |

ARC-E was dropped on 2026-09-04 (see below), so the diagnostic rests on ARC-C
for the long-generation side.

Table note this needs: PIQA is not in LLaDA's published Instruct results and is
our own added control.

**The dissociation reading stays a hypothesis until ARC-C's 20% cells land.**
ARC-C is the column that separates "content is commonsense" from "protocol is
long generation": at 40% reduction Clean 54.2 -> Traj 65.3, a paired +11.2pp
(95% CI [+8.3, +14.1], McNemar p = 7e-14, 218 vs 88 discordant), which is
larger than the GSM8K gap at the same ratio, on a benchmark whose *content* is
grade-school science. ARC-E was to have been the replication; with it dropped, the
reading now stands or falls on ARC-C's own 20% cells alone, which is a weaker
basis -- one benchmark rather than two -- and the text should say so rather
than generalise from a single column.

## Benchmarks deliberately not run

Recorded so the decision is not silently revisited:

- **GPQA** — an 8B dense model scores near chance (~26-30%); every compressed
  arm would sit at the floor and the column could not separate calibrations.
- **MMLU-Pro** — 12,032 questions at 256 steps each; not feasible before the
  deadline at any useful arm count.
- **ARC-C at the 3-token protocol** — would disagree with the official ARC-C
  config the main run uses. Held as the fallback experiment if a reviewer
  challenges the long-vs-short-generation attribution, not run now.
- **HumanEval / MBPP** — every compressed arm sat at the floor (0-12%), so the
  columns carried no signal. Results stay on disk.
- **IFEval / BBH** — called off on cost; caches stay on disk and resume.
- **ARC-E** — started at the full 2365, then cut to a pre-registered 800-item
  subset, then dropped entirely on 2026-09-04 before any cell finished. No
  ARC-E number was ever scored, so nothing was selected on. The subset manifest
  (`results/eval_stageB/arc_e_subset_manifest.json`), the generation caches and
  `oc_tasks.data.arc_e_subset` all stay on disk; the column is simply not run.
- No further benchmarks are added beyond this scope.

## Resolved: the official MATH sampler, and which family each column is in

Settled against LLaDA's own `evaluation/EVAL.md` (local copy at
`/zpool-00/home/tl356/LLaDA/evaluation/EVAL.md`). There was no contradiction:
the two records in this repo are two *different* official settings, and the
third figure is from a different model.

LLaDA-8B-Instruct is published under two sampling families:

| Family | Setting | Official (OpenCompass) |
| --- | --- | --- |
| pure diffusion (paper Tab. 1/2) | MATH gen 512 / block 512, `confidence_eos_eot_inf=True` | 29.6 |
| block diffusion (EOS mitigation) | MATH gen 512 / **block 64**, both eos flags False | **42.7** |

`trajmc/prompts.py::OFFICIAL_SAMPLER['math']` records the first;
"The rollout sampler configuration" above records the second. `gen 256 /
block 256` for Math appears only in the **LLaDA-8B-Base** lm-eval sweep (30.3),
not in any Instruct table, so it is not an Instruct protocol at all.

Every column we ran matches an official Instruct setting exactly, and every
dense number lands within 1.3pp of the published one:

| Column | Our sampler | Family | Dense | Official | Δ |
| --- | --- | --- | ---: | ---: | ---: |
| MMLU | gen 3 / block 3 | pure | 65.3 | 65.4 | −0.1 |
| HellaSwag | gen 3 / block 3 | pure | 76.6 | 75.3 | +1.3 |
| ARC-C | gen 512 / block 512 | pure | *pending* | 89.2 | — |
| MATH-500 | gen 512 / block 512, C=True | pure | 30.2 | 29.6 | +0.6 |
| GSM8K | gen 256 / block 8 | **block** | 80.2 | 78.9 | +1.3 |

### What the MATH-500 column can carry on its own

Paired, on the finished pure-diffusion column (per-item records, no new compute):

| Pair, 20% reduction | gap | 95% CI | McNemar p | discordant | significant |
| --- | ---: | ---: | ---: | ---: | --- |
| MATH-500 Clean 9.0 -> Traj 11.0 | +2.0pp | [-0.6, +4.8] | 0.193 | 29 / 19 | **no** |
| GSM8K Clean 42.2 -> Traj 56.7 | +14.5pp | [+11.5, +17.4] | 5e-21 | 307 / 116 | yes |

MATH-500 sits near the floor for every compressed arm (9-11%), so the effect
there is too small to reach significance on its own. **The column's role is
supporting evidence whose direction agrees with GSM8K, not an independent
result**, and the text should present it that way. It also bounds what the
family choice can cost: a column that is not independently significant under
either family cannot have the paper's conclusion turn on which family it uses.

**The live question is therefore not 512-vs-256 but which family MATH-500 sits
in.** GSM8K is on block diffusion and MATH-500 is on pure diffusion; both are
official, but the same model scores 29.6 and 42.7 on Math depending on which is
chosen. Internal Clean/Traj/Actual comparisons are unaffected either way (every
arm shares one sampler); the choice only sets what the column can be compared
against externally, and whether the two reasoning columns are described as
coming from one protocol or two.

`scripts/run_math500_pilot.sh` measures it on 300 fixed problems, three arms,
block-diffusion side only (the pure side is sliced out of the finished runs).
The decision rule and its one amendment are in
`results/pilot_math500/items_manifest.json`, both timestamped as fixed before
any block-diffusion answer was generated. The amendment replaced a bare
sign test with a significance test: SE(gap) at n=300 is about 2.2pp against a
true gap of about 2pp, so the sign alone would have triggered a ~117 GPU-h
column rerun on noise.

