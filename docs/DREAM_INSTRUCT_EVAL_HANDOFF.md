# Handoff: Dream-7B-Instruct evaluation on GSM8K / MATH-500 / GPQA

You are running three evaluation columns of a compression study on
**Dream-v0-Instruct-7B**. Everything below is fixed; the point of this handoff
is that the numbers must be produced under *exactly* this protocol, because
they will be compared cell-to-cell against runs done elsewhere. **Do not
"improve" any setting.** If something does not work, say so rather than
substituting an alternative.

---

## 1. What you are producing

A 3-column × 7-row table. Each row is one model checkpoint ("cell"), each
column one benchmark.

| Cell | What it is | Weights |
| --- | --- | --- |
| `dense` | the unmodified model | the HF checkpoint |
| `r08_weight_svd` | 20% param reduction, plain weight SVD | A/B factor directory |
| `r08_clean` | 20%, activation SVD, clean-text calibration | A/B factor directory |
| `r08_traj` | 20%, activation SVD, trajectory calibration | A/B factor directory |
| `r06_weight_svd` | 40% reduction, plain weight SVD | A/B factor directory |
| `r06_clean` | 40%, clean calibration | A/B factor directory |
| `r06_traj` | 40%, trajectory calibration | A/B factor directory |

Columns: `gsm8k_cot`, `minerva_math500`, `gpqa_main_n_shot`. **21 runs total.**

---

## 2. What you need copied over

| Item | Size | Note |
| --- | --- | --- |
| `baselines/dream_eval_instruct/` | ~3 MB | the evaluation harness — see §3 |
| Dream-v0-Instruct-7B checkpoint | 16 GB | or pull `Dream-org/Dream-v0-Instruct-7B` from HF |
| 6 × A/B factor directories | ~30 GB each, **~180 GB total** | only needed for the 6 compressed cells |

Each factor directory holds 392 files named
`model_layers_<i>_<proj>_A.pt` / `_B.pt` plus a `compression_summary.json`.
Copy them whole; a partial directory will be detected and refused.

If 180 GB is impractical — and it usually is — **build the factors locally
instead**: the calibration tensors they are derived from are 19 MB and ship
with this repo. See §2b. Transferring 180 GB will typically take longer than
recompressing.

Either way `dense` needs no factors at all, so it can start immediately.

---

## 2b. Building the compressed cells yourself (recommended)

The expensive input to compression is the calibration tensor, and those are
already in the repo at
`results/stage_a/dream_instruct_r0.8/calib/` (19 MB, three files):

| File | Scheme | Feeds arm |
| --- | --- | --- |
| `dream_instruct_clean_t0_c4_n256_s42_full_8ff961b_calib.pt` | clean text, t = 0 | `*_clean` |
| `dream_instruct_random_t_c4_n256_s42_full_8ff961b_calib.pt` | one t ~ U(0,1) per window | `*_traj` |
| `dream_instruct_grid_t_prefix_p0p25_c4_n256_s42_full_8ff961b_calib.pt` | stratified grid + 25% visible prefix | `*_mcs` — **not in this table**, ignore unless asked |

Each is 256 windows of 2048 C4 tokens, seed 42. `*_weight_svd` needs no
calibration at all (it is a plain SVD of the weight matrix).

Do **not** regenerate them. They are the shared input every arm is matched on;
a fresh draw would silently break that matching. If you must know they are
intact, `results/stage_a/dream_instruct_r0.8/reports/calibration_audit.json`
records the audit that every arm shares byte-identical pre-noise windows.

### Commands

Six runs, one per cell. `--ratio` is the parameter **retention** fraction, so
0.8 is the 20%-reduction row and 0.6 the 40% one.

```bash
CALIB=results/stage_a/dream_instruct_r0.8/calib
MODEL=/path/to/Dream-v0-Instruct-7B

for RATIO in 0.8 0.6; do
  TAG=r0${RATIO#0.}          # 0.8 -> r08, 0.6 -> r06

  # weight_svd: no calibration, plain SVD of W
  python -m trajmc.compression --backend dream_instruct --model_path "$MODEL" \
    --ratio "$RATIO" --layer_type all --decomp identity \
    --save_path "weights/${TAG}_weight_svd" \
    --save_dtype bfloat16 --linalg_device cuda --batch_size 1 \
    --run_id "${TAG}_weight_svd"

  # clean and traj: activation-weighted SVD, one calibration each
  for ARM in clean traj; do
    case $ARM in
      clean) C="$CALIB/dream_instruct_clean_t0_c4_n256_s42_full_8ff961b_calib.pt" ;;
      traj)  C="$CALIB/dream_instruct_random_t_c4_n256_s42_full_8ff961b_calib.pt" ;;
    esac
    python -m trajmc.compression --backend dream_instruct --model_path "$MODEL" \
      --ratio "$RATIO" --layer_type all --decomp cholesky --calib "$C" \
      --save_path "weights/${TAG}_${ARM}" \
      --save_dtype bfloat16 --linalg_device cuda --batch_size 1 \
      --run_id "${TAG}_${ARM}"
  done
done
```

Point `weights_path=` at these `weights/<cell>` directories when you run the
evaluation.

### What a finished cell looks like

392 factor files (`model_layers_<i>_<proj>_A.pt` / `_B.pt`, 196 Linears × 2)
plus `compression_summary.json`. Check the summary before trusting a cell:

```
n_target_linears  196     # 28 decoder layers x 7 projections
n_compressed      196     # every one of them replaced
kept_fraction     ~=ratio # measured, e.g. 0.5999 for --ratio 0.6
factor_file_dtype bfloat16
```

`n_compressed` below 196, or a missing summary, means the run did not finish —
delete the directory and rerun rather than evaluating a half-written cell.
Embedding and `lm_head` are never compressed; that is deliberate and matches
SVD-LLM / ASVD practice.

### Cost, measured on one L40S

**About 3 hours per cell**, 6 cells ≈ 18 GPU-hours, and they are fully
independent so run them in parallel. Each cell writes ~30 GB, so budget
~180 GB of disk. Peak host RAM was 55 GB; 110 GB is a safe request.

Two traps we hit:

- **Do not set a wall limit under 4 hours.** Our first batch was submitted with
  90 minutes and all eight arms were killed at 1:30:13 with the output
  directories 80% written.
- **A half-written directory is not detected by the compressor on rerun.** Wipe
  `weights/<cell>` before restarting it, or the new run mixes with the old
  files.

---

## 3. The harness — read this before installing anything

`baselines/dream_eval_instruct/` is the **official** `eval_instruct/` directory
from <https://github.com/DreamLM/Dream> (vendored at commit
`31f94a60d187e3fd481fee3bbc2c732eb94a879c`). It is its own fork of
lm-evaluation-harness — **not** the same as the `eval/` directory in that repo,
which is the *Base* model protocol. The Instruct protocol differs in model
class (`diffllm`, not `dream`), in task configs, and in applying a chat
template to every task.

It carries exactly two additions, both marked `[trajmc]` and both purely
additive (zero upstream lines deleted):

1. `lm_eval/models/diffllm.py` — a `weights_path` model arg that swaps the 196
   target Linear layers for their rank-k A/B factorisation right after the
   dense load. Nothing else in the generation or likelihood path is touched, so
   `dense` and every compressed cell run the identical code.
2. `lm_eval/tasks/trajmc_added/minerva_math500.yaml` — the official
   `minerva_math` task pointed at `HuggingFaceH4/MATH-500` instead of the full
   MATH test set. Prompt construction, few-shot block and grading all come from
   the official `lm_eval/tasks/minerva_math/utils.py`, unchanged.

Run it with `PYTHONPATH` pointing at that directory; do **not** `pip install`
it over an existing lm-eval unless you want it to become the global one.

### Environment

- **`transformers >= 4.46` is mandatory.** Dream's checkpoint records 4.46.2 in
  its `config.json` and its vendored modeling code forwards a `weights_only`
  argument that older versions do not accept. On 4.45.x the load dies with
  `TypeError: DreamModel.__init__() got an unexpected keyword argument
  'weights_only'`. Do not work around this by stripping the argument — the
  forward pass is what produces every number here. Use a conforming
  environment. (Verified working: `transformers 4.49.0`, `torch 2.10`,
  `datasets 4.8.4`, `accelerate 0.34.2`.)
- Also needed: `lm_eval`'s deps plus `math_verify`, `sympy`, `antlr4-python3-runtime`.
- **GPQA is a gated dataset.** Accept the terms at
  <https://huggingface.co/datasets/Idavidrein/gpqa> with your HF account, then
  export `HF_TOKEN`. Without it `load_dataset` raises `DatasetNotFoundError`.

---

## 4. The exact commands

Set these once:

```bash
export HARNESS=/path/to/baselines/dream_eval_instruct
export MODEL=/path/to/Dream-v0-Instruct-7B
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # needed for GPQA
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="$HARNESS"
```

`ARGS` differs per column. For a **compressed** cell append
`,weights_path=/path/to/that/cells/factor/dir` to `ARGS`; for `dense` append
nothing.

### GSM8K

```bash
python -m accelerate.commands.launch --num_processes 1 -m lm_eval \
  --model diffllm \
  --model_args "pretrained=$MODEL,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy" \
  --tasks gsm8k_cot --device cuda --batch_size 1 --num_fewshot 0 \
  --output_path out/<cell>/gsm8k_cot \
  --log_samples --confirm_run_unsafe_code --apply_chat_template
```

### MATH-500

```bash
python -m accelerate.commands.launch --num_processes 1 -m lm_eval \
  --model diffllm \
  --model_args "pretrained=$MODEL,trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy" \
  --tasks minerva_math500 --device cuda --batch_size 1 --num_fewshot 0 \
  --output_path out/<cell>/minerva_math500 \
  --log_samples --confirm_run_unsafe_code --apply_chat_template
```

### GPQA

Note this one passes **no sampler arguments at all** — that is the official
row, and it is a likelihood task, so they would be unused. Do not add them for
consistency's sake.

```bash
python -m accelerate.commands.launch --num_processes 1 -m lm_eval \
  --model diffllm \
  --model_args "pretrained=$MODEL,trust_remote_code=True,dtype=bfloat16" \
  --tasks gpqa_main_n_shot --device cuda --batch_size 1 --num_fewshot 5 \
  --output_path out/<cell>/gpqa_main_n_shot \
  --log_samples --confirm_run_unsafe_code --apply_chat_template
```

`--num_processes N` with `N` GPUs shards the request list and gathers it; the
protocol and per-item results are unchanged, only wall clock. Use it freely.

---

## 5. Do the dense cell first and check it

Run all three columns on `dense` before touching the compressed cells, and
compare against **Dream 7B Instruct, Table 2 of arXiv:2508.15487**:

| Column | Paper | Note |
| --- | --- | --- |
| GSM8K | **81.0** | directly comparable |
| GPQA | **33.0** | directly comparable |
| MATH | 39.2 | **not** directly comparable — the paper's cell is full MATH (~5000 problems); we deliberately run the 500-problem MATH-500 subset, so expect a different value |

Gate: within ~1 point is fine. **More than 5 points off on GSM8K or GPQA means
something is wrong with the setup — stop and report it rather than continuing
to the compressed cells.**

With `--log_samples` on, also eyeball 10 samples per column and confirm:
the chat template is present, the few-shot block is not mangled by it,
generations are not truncated at the length cap, and no output is empty. Report
the fraction of generations that hit the length limit.

---

## 6. Where results land, and what to send back

lm-eval writes to **`<output_path>/<sanitised model name>/results_*.json`** —
one directory deeper than `--output_path`. A checker that globs
`<output_path>/results_*.json` will wrongly conclude the run failed; ours did,
which is why this is called out.

Send back, per cell and column:

- the whole `results_*.json`
- the `samples_*.jsonl` (needed for paired significance tests between cells —
  aggregate accuracy alone cannot tell a real gap from noise)
- wall-clock time and the GPU model used

Headline metrics: `exact_match` for GSM8K and MATH-500, `acc` for GPQA.

---

## 7. Measured cost, so you can plan

On one L40S, batch size 1, uncontended. Multiply by ~2.5 on an A5000, and by
up to 3 if another process shares the card.

| Column | Items | Per cell | 7 cells |
| --- | --- | ---: | ---: |
| GSM8K | 1319 | 4.4–8 h | 30–56 h |
| MATH-500 | 500 | 5.1–6.2 h | 36–43 h |
| GPQA | 448 | short (likelihood, not generation) | — |

These are single-GPU figures; `--num_processes N` divides the wall clock.

---

## 8. Traps we already hit — do not repeat them

1. **Check your card is not shared.** A card with another process on it ran
   2.7x slower (244 s/question against 89 s alone) with no error and no warning
   — it looks like "this benchmark is slow", not like a fault. Before trusting
   a timing, run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
   on the device you were given.
2. **Never pass a comma-containing value through `sbatch --export`.** It splits
   on commas, so `SAMPLER="temperature=0.1,top_p=0.9,alg=entropy"` silently
   becomes `temperature=0.1` and the run is on the wrong sampler. This cost us
   about 11 GPU-hours before it was caught. Encode it, or set the variable in
   the submitting shell and use `--export=ALL`.
3. **`weights_path` pointing at an empty or partial directory** raises rather
   than silently producing dense numbers — that guard is deliberate, do not
   relax it.
4. Results land one directory deeper than `--output_path` (see §6).
