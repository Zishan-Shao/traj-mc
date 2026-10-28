# Traj-MC: Trajectory-Aware Low-Rank Compression for Diffusion Language Models

Code for *Preserving Mathematical Reasoning in Compressed Diffusion Language
Models via Trajectory-Aware Low-Rank Approximation* (MATH-AI Workshop,
NeurIPS 2026).

Activation-aware low-rank methods such as SVD-LLM calibrate on clean, fully
visible text, but a diffusion LM spends inference on partially masked states.
Traj-MC estimates the second moment that defines the low-rank objective from
trajectory states instead: for each calibration window it samples a corruption
level `t ~ U(0, 1)` and masks every token independently with probability `t`,
accumulates `Sigma_N = (1/N) sum h h^T`, and applies the whitened SVD-LLM
solution `(W S_N)_k S_N^{-1}`. Compressed layers and rank allocation are
unchanged; only the calibration states differ.

## Paper names and code names

Every compressed model is built in two steps: a calibration set
(`trajmc.calibration --scheme ...`) and a compression run
(`trajmc.compression --decomp ...`). The code names of the arms predate the
paper, so they differ from the paper's names. This table maps them.

| Paper (Table 1 / figures) | Calibration `--scheme` | Compression | Weights / eval directory |
|---|---|---|---|
| SVD (no activations) | none | `--decomp identity` | `weight_svd` |
| SVD-LLM (clean, `t = 0`) | `clean_t0` | `--decomp cholesky` (LLaDA) or `eigh` (Dream) | `clean` |
| +Traj-MC (`t ~ U(0,1)`) | `random_t` | same as SVD-LLM | `traj` |
| Fixed-t control (Fig. 2a) | `fixed_t --fixed_t T` | same as SVD-LLM | `fixed_t0p25`, `fixed_t0p5`, `fixed_t0p75` |
| Held-out generation states (E_gen) | `rollout --rollout_prompts gsm8k_test` | not compressed | passed to E_gen as `--states` |

- **`random_t` is Traj-MC.** The name means that every window draws its own
  random corruption level `t ~ U(0, 1)`. **`fixed_t`** is the Fig. 2a control:
  the same masking, but every window uses one given `t`.
- On Dream, the weights directories carry the layer scope as a prefix:
  `nokv_weight_svd`, `nokv_clean`, `nokv_traj`.
- `--decomp identity` means identity whitening (`S = I`), so the result is a
  plain truncated SVD of `W`.
- `rollout` records real states from the dense model's reverse sampler. The
  paper uses them only to *measure* E_gen; no reported model is calibrated on
  them. Code comments call this the "Actual-State" control.
- `--arm base` and `--arm ours` are older aliases for `--scheme clean_t0` and
  `--scheme random_t`.
- `grid_t` and `grid_t_prefix` (a deterministic grid `t = i/N`, the second with
  a visible prefix) are kept for completeness. They are not used in the paper.
- In the LLaDA evaluation harness, `--mode uncompressed` is the dense model and
  `--mode traject_svd` loads the low-rank factors written by
  `trajmc.compression`. The harness's other `--mode` values come from earlier
  work and are not used in the paper.
- Labels inside the analysis reports follow the same names: E_gen reports use
  `weight_svd`, `clean`, `traj`, `fixed_t0p25`, ...; the Fig. 2b report uses
  `random_t_n{N}`, `clean_t0_n256`, and `independent_random_t` for the
  reference draw.

### Where the method lives

| Step | Function |
|---|---|
| Clean C4 windows, shared by every arm | `trajmc/calibration.py: build_windows_streaming` |
| Traj-MC masking (`random_t`) | `trajmc/calibration.py: apply_noise` |
| Fixed-t masking (`fixed_t`) | `trajmc/sampling.py: apply_forward_mask` |
| Held-out generation states (`rollout`) | `trajmc/sampling.py: llada_block_rollout_states` |
| `Sigma_N` accumulation (forward hooks) | `trajmc/compression.py: collect_xtx_gpu` |
| Whitened truncated SVD, `(W S)_k S^{-1}` | `trajmc/compression.py: whiten_truncate` |
| Rank from the kept-parameter ratio | `trajmc/common.py: rank_from_ratio` |

## Layout

```
trajmc/                core method
  calibration.py         builds a calibration set for one --scheme from 256 shared
                         C4 windows; apply_noise is the Traj-MC (random_t) masking
  sampling.py            scheme definitions, apply_forward_mask (fixed t / grid),
                         and reverse-sampler state capture for --scheme rollout
  compression.py         second-moment accumulation and whitened truncated SVD
                         (whiten_truncate); writes <layer>_A.pt / <layer>_B.pt factors
  common.py              backends (LLaDA, Dream), model loading, target layers
  prompts.py             LLaDA's official GSM8K/MATH prompts for --scheme rollout
  analysis/
    gen_reconstruction.py    E_gen on held-out generation states (Fig. 1b, 2a)
    covariance_estimation.py Sigma_N error against an independent draw (Fig. 2b)
    subspace_distance.py     projector distance between kept subspaces
    calibration_ablation.py  audit: identical clean windows across all arms
scripts/               bash scripts that reproduce the paper's experiments
utils/                 evaluation code
  eval_benchmarks.py     LLaDA benchmark harness (--mode uncompressed | traject_svd)
  generate.py            LLaDA low-confidence block sampler
  oc_tasks/              dataset loaders and answer graders (OpenCompass protocols)
  dream_eval_instruct/   Dream's official Instruct evaluation toolkit (lm-eval fork)
  collect_results.py     prints Table 1 and the figure numbers from results/
tests/                 unit tests (pytest)
```

## Installation

```bash
pip install -e ".[eval,test]"
pip install -e "utils/dream_eval_instruct[ifeval,math]"   # only for the Dream evaluation
```

We used Python 3.10. Calibration and compression ran with torch 2.10 and
transformers 4.49. The LLaDA benchmarks ran with torch 2.1.2 and
transformers 4.45.2. All steps run on a single GPU. A 24 GB card is enough
for LLaDA. Dream compression needs about 32 GB with `LINALG_DEVICE=cuda`. On a
smaller card, use `LINALG_DEVICE=cpu`.

Model paths default to the Hugging Face IDs (`GSAI-ML/LLaDA-8B-Instruct`,
`Dream-org/Dream-v0-Instruct-7B`). To use local copies, set `LLADA_PATH` and
`DREAM_PATH`. Set `PYTHON_BIN` to choose the interpreter. All outputs are
written under `results/`.

## Reproducing the paper

| Paper | Script | Output |
|---|---|---|
| Table 1, LLaDA-8B-Instruct, compression | `scripts/llada_compress.sh` | `results/llada_instruct/r{0.8,0.6}/weights/{weight_svd,clean,traj}/` |
| Table 1, LLaDA-8B-Instruct, benchmarks | `scripts/llada_eval.sh` | `results/llada_instruct/eval/` |
| Table 1, Dream-7B-Instruct, compression | `scripts/dream_compress.sh` | `results/dream_instruct/r0.8/weights/nokv_{weight_svd,clean,traj}/` |
| Table 1, Dream-7B-Instruct, benchmarks | `scripts/dream_eval.sh` | `results/dream_instruct/eval/` |
| Fig. 1b, E_gen on held-out generation states | `scripts/llada_compress.sh` (measure stage) | `results/llada_instruct/reports/gen_reconstruction_r0.8_*.json` |
| Fig. 1c, reasoning retention | the LLaDA 20% cells of Table 1 | `results/llada_instruct/eval/r0.8_{clean,traj}/` |
| Fig. 2a, fixed t vs. t ~ U(0,1) | `scripts/fixed_t.sh` (the `t = 0` point is the `clean` arm) | `results/llada_instruct/reports/fixed_t_r0.8_*.json` |
| Fig. 2b, Monte Carlo convergence | `scripts/mc_budget.sh` | `results/llada_instruct/reports/mc_budget_covariance.json` |

```bash
bash scripts/llada_compress.sh      # calibrate, compress 20% and 40%, measure E_gen
bash scripts/llada_eval.sh          # GSM8K, MATH-500, SVAMP, ARC-C, MMLU, HellaSwag, PIQA
bash scripts/fixed_t.sh             # needs llada_compress.sh
bash scripts/mc_budget.sh           # needs llada_compress.sh
bash scripts/dream_compress.sh
bash scripts/dream_eval.sh
python utils/collect_results.py     # print every table cell and figure value
```

Each script skips any step whose output already exists, so an interrupted run
can be restarted. The compression scripts also take `STAGE=calibrate|compress|measure`.
The evaluation scripts take `CELLS` and `BENCHMARKS` (LLaDA) or `TASKS` (Dream)
to run a subset, e.g. `CELLS="r0.8/traj" BENCHMARKS="gsm8k" bash scripts/llada_eval.sh`.

### One arm by hand

The scripts chain three commands. For example, the 20% Traj-MC row for LLaDA
is built and evaluated like this:

```bash
# 1. calibration set: 256 C4 windows, each masked at its own t ~ U(0,1)
python -m trajmc.calibration --backend llada_instruct --scheme random_t \
    --corpus c4 --c4_streaming --nsamples 256 --seqlen 2048 \
    --seed 42 --sampling_seed 42 --out_dir results/llada_instruct/calib
# 2. whitened truncated SVD, 80% of parameters kept
python -m trajmc.compression --backend llada_instruct --ratio 0.8 \
    --layer_type all --decomp cholesky --batch_size 1 --save_dtype bfloat16 \
    --calib results/llada_instruct/calib/llada_instruct_random_t_c4_n256_s42_full_<git>_calib.pt \
    --save_path results/llada_instruct/r0.8/weights/traj
# 3. benchmark (batch size 1, as in the paper)
python utils/eval_benchmarks.py --mode traject_svd \
    --weights_path results/llada_instruct/r0.8/weights/traj \
    --benchmarks gsm8k --batch_size 1
```

Use `--scheme clean_t0` for the SVD-LLM row. Use `--decomp identity` without
`--calib` for the SVD row. The calibration file name contains the current git
hash. Add `--print_artifact_path` to step 1 to print it without building
anything.

### Settings

- **Calibration.** Every activation-aware arm uses the same 256 C4 windows of
  2048 tokens (window seed 42). A byte-level audit
  (`trajmc.analysis.calibration_ablation`) checks that the windows are
  identical across arms before compression. Only the masking differs.
- **Compression.** `RATIO` is the retained parameter fraction. 0.8 means 20%
  reduction and 0.6 means 40%. All arms of a model use the same layers and the
  same rank allocation.
- **LLaDA** compresses all seven linear layers in every block and uses a
  Cholesky factor of `Sigma_N`.
- **Dream** keeps the GQA `k_proj`/`v_proj` dense (`LAYER_TYPE=nokv`: 140
  linears, 80.3% of parameters kept). Its MLP second moments are numerically
  singular in float32, so it uses an eigendecomposition with a relative
  eigenvalue floor (`--decomp eigh --rcond 1e-10`).
- **Evaluation.** Each model follows its official protocol.
  - *LLaDA* uses the OpenCompass configurations from the LLaDA repository,
    with generative evaluation and the low-confidence block sampler.
  - *Dream* uses its Instruct toolkit with the chat template, temperature 0.1,
    top-p 0.9, and the entropy remasking order. GSM8K is scored on the first
    512 test items, and MMLU on the first 40 items of each subject.
    HellaSwag, PIQA and ARC-C use Dream's Base multiple-choice protocol.
    SVAMP uses the GSM8K row's prompt and sampler settings, on all test items.
- **Second-moment error (Fig. 2b).** Relative Frobenius error of `Sigma_N`
  against an independent 256-window Traj-MC draw (window seed 31337),
  averaged over three probe layers (`blocks.0.attn_out`, `blocks.16.attn_out`,
  `blocks.31.up_proj`). Each moment is accumulated exactly on a fixed set of
  256 randomly chosen input coordinates per layer (`--sketch_dim 256`,
  `--sketch_seed 2027`), not on the full `d x d` matrix.
- **E_gen.** Measured on real reverse-sampler states of the dense model for
  256 held-out GSM8K test prompts (seed 1337, 4-shot, generation length 256,
  block 8, 256 steps). Each prompt contributes one intermediate state, and
  the 256 states tile sampler steps 0-255. Error is restricted to the
  generation region. Confidence intervals come from 2000 paired bootstrap
  resamples over prompts.

## Tests

```bash
pytest tests
```

Set `LLADA_INSTRUCT_PATH` to a local LLaDA-8B-Instruct checkout to also run the
prompt-construction tests.

## License

MIT (see `LICENSE`). Third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`.
