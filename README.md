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

## Layout

```
trajmc/            core method
  calibration.py     builds calibration windows: clean (t = 0), Traj-MC (t ~ U(0,1)),
                     fixed t, and held-out generation states from the dense sampler
  sampling.py        masking operator and diffusion-sampler state capture
  compression.py     second-moment accumulation and whitened truncated SVD
  common.py          model loading, backends (LLaDA, Dream), target layers
  analysis/          measurements: E_gen, second-moment error, subspace distance,
                     and the calibration audit (identical windows across arms)
scripts/           bash scripts that reproduce the paper's experiments
utils/             shared evaluation code
  eval_benchmarks.py LLaDA benchmark harness (LLaDA's OpenCompass protocols)
  generate.py        LLaDA low-confidence block sampler
  oc_tasks/          dataset loaders and answer graders
  dream_eval_instruct/  Dream's official Instruct evaluation toolkit (lm-eval fork)
  collect_results.py prints Table 1 and the figure numbers from results/
tests/             unit tests (pytest)
```

## Installation

```bash
pip install -e ".[eval,test]"
pip install -e "utils/dream_eval_instruct[ifeval,math]"   # only for the Dream evaluation
```

We used Python 3.10. Calibration and compression ran with torch 2.10 and
transformers 4.49; the LLaDA benchmarks ran with torch 2.1.2 and
transformers 4.45.2. All steps run on a single GPU. A 24 GB card is
enough for LLaDA. Dream compression needs about 32 GB with
`LINALG_DEVICE=cuda`; on a smaller card, use `LINALG_DEVICE=cpu`.

Model paths default to the Hugging Face IDs (`GSAI-ML/LLaDA-8B-Instruct`,
`Dream-org/Dream-v0-Instruct-7B`). To use local copies, set `LLADA_PATH` and
`DREAM_PATH`. Set `PYTHON_BIN` to choose the interpreter. All outputs are
written under `results/`.

## Reproducing the paper

| Paper | Script | Output |
|---|---|---|
| Table 1, LLaDA-8B-Instruct, compression | `scripts/llada_compress.sh` | `results/llada_instruct/r{0.8,0.6}/weights/` |
| Table 1, LLaDA-8B-Instruct, benchmarks | `scripts/llada_eval.sh` | `results/llada_instruct/eval/` |
| Table 1, Dream-7B-Instruct, compression | `scripts/dream_compress.sh` | `results/dream_instruct/r0.8/weights/` |
| Table 1, Dream-7B-Instruct, benchmarks | `scripts/dream_eval.sh` | `results/dream_instruct/eval/` |
| Fig. 1b, E_gen on held-out generation states | `scripts/llada_compress.sh` (measure stage) | `results/llada_instruct/reports/gen_reconstruction_r0.8_*.json` |
| Fig. 2a, fixed t vs. t ~ U(0,1) | `scripts/fixed_t.sh` | `results/llada_instruct/reports/fixed_t_r0.8_*.json` |
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

### Settings

- **Calibration.** Every activation-aware arm uses the same 256 C4 windows of
  2048 tokens (window seed 42). A byte-level audit checks that the windows are
  identical across arms before compression; only the masking differs.
- **Compression.** `RATIO` is the retained parameter fraction. 0.8 means 20%
  reduction and 0.6 means 40%. All arms of a model use the same layers and the
  same rank allocation. *SVD* is a plain truncated SVD of `W`.
  *SVD-LLM* is whitening on clean windows. *+Traj-MC* is whitening on
  trajectory windows.
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
    SVAMP uses the GSM8K row.
- **Second-moment error (Fig. 2b).** Relative Frobenius error of `Sigma_N`
  against an independent 256-window Traj-MC draw (window seed 31337),
  averaged over three probe layers (`blocks.0.attn_out`, `blocks.16.attn_out`,
  `blocks.31.up_proj`). Each moment is accumulated exactly on a fixed set of
  256 randomly chosen input coordinates per layer (`--sketch_dim 256`,
  `--sketch_seed 2027`), not on the full `d x d` matrix.
- **E_gen.** Measured on real reverse-sampler states of the dense model for
  256 held-out GSM8K test prompts (seed 1337, 4-shot, generation length 256,
  block 8, 256 steps), restricted to the generation region. Confidence
  intervals come from 2000 paired bootstrap resamples over prompts.

## Tests

```bash
pytest tests
```

Set `LLADA_INSTRUCT_PATH` to a local LLaDA-8B-Instruct checkout to also run the
prompt-construction tests.

## License

MIT (see `LICENSE`). Third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`.
