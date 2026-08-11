# Traj-MC

Core implementation of trajectory-matched calibration (Traj-MC) for masked
diffusion language models. The repository contains the current random-`t`
method for both LLaDA and Dream in one `trajmc` package.

No model checkpoints, calibration tensors, datasets, raw evaluation outputs,
or generated low-rank factors are included.

## Method

Traj-MC changes only the calibration inputs used by activation-aware low-rank
compression:

- `base`: clean token windows (`t = 0`);
- `ours`: sample one `t ~ Uniform(0, 1)` per window and independently replace
  each token with the model's mask token with probability `t`.

Given the same seed, both arms use byte-identical clean windows before masking.
The covariance collection, whitening, rank formula, and truncation path are
shared, making the comparison single-variable.

## Supported backends

| Backend | Default model | Mask ID | Compressed Linear layers |
| --- | --- | ---: | ---: |
| `llada` | `GSAI-ML/LLaDA-8B-Base` | 126336 | 224 (32 × 7) |
| `dream` | `Dream-org/Dream-v0-Base-7B` | 151666 | 196 (28 × 7) |

The embedding and output head always remain dense. A model-graph check aborts
if the expected backend layout has changed.

## Install

Python 3.10+ and a CUDA-enabled PyTorch environment are recommended.

```bash
pip install -e .
pip install -e '.[eval]'  # only when running lm-eval
```

Remote model code is loaded through Hugging Face Transformers. You can pass a
local checkpoint directory with `--model_path` to every command.

## 1. Build paired calibration sets

Use the same backend, corpus, seed, sequence length, and sample count for both
arms:

```bash
trajmc-calibrate \
  --backend llada --arm base --corpus c4 --c4_streaming \
  --nsamples 256 --seqlen 2048 --seed 42

trajmc-calibrate \
  --backend llada --arm ours --corpus c4 --c4_streaming \
  --nsamples 256 --seqlen 2048 --seed 42
```

Replace `llada` with `dream` to run the same random-`t` construction on Dream.
Calibration artifacts are written under `results/calib/<backend>/` by default.

## 2. Compress

Run the identical command for the two calibration files, changing only
`--calib` and `--save_path`:

```bash
trajmc-compress \
  --backend llada \
  --calib results/calib/llada/<base-calibration>.pt \
  --ratio 0.8 --layer_type all --decomp cholesky \
  --save_path results/weights/llada/base

trajmc-compress \
  --backend llada \
  --calib results/calib/llada/<ours-calibration>.pt \
  --ratio 0.8 --layer_type all --decomp cholesky \
  --save_path results/weights/llada/ours
```

`ratio` is the target retained parameter fraction:

```text
k = int(ratio * out_features * in_features / (out_features + in_features))
```

The output contains per-layer `A`/`B` factors and a
`compression_summary.json`. All generated artifacts live under ignored paths.

Verify that BASE and OURS differ only in calibration identity/runtime fields:

```bash
python -m trajmc.analysis.summary_diff \
  --base results/weights/llada/base \
  --ours results/weights/llada/ours
```

## 3. Evaluate

The package includes the LLaDA and Dream lm-eval adapters used by the current
experiments:

```bash
trajmc-evaluate \
  --backend llada --task piqa --arm ours \
  --weights results/weights/llada/ours

trajmc-evaluate \
  --backend dream --task gsm8k_cot --arm ref
```

Add `--dry` to inspect the generated `accelerate` command without running it.
Supported tasks and their fixed per-backend protocols live in
`trajmc/common.py`.

To produce paired records and run McNemar's test:

```bash
trajmc-items --backend llada --task piqa --arm base
trajmc-items --backend llada --task piqa --arm ours

python -m trajmc.analysis.mcnemar \
  --base_items <base-items.jsonl> \
  --ours_items <ours-items.jsonl> \
  --benchmark piqa
```

## Repository layout

```text
trajmc/
├── calibration.py          # shared clean-t0 / random-t construction
├── compression.py          # shared covariance + whitening + truncation
├── common.py               # LLaDA/Dream backend specifications
├── evaluate.py             # one evaluation launcher for both backends
├── eval/                   # architecture-specific official lm-eval adapters
└── analysis/               # paired-item conversion and statistical gates
tests/                      # CPU-only core tests
```

The evaluation adapters retain attribution comments to their upstream LLaDA,
Sink-Aware, and Dream sources. Model weights and datasets remain governed by
their respective upstream licenses.
