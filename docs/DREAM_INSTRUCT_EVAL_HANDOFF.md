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

If 180 GB is impractical, run `dense` first and ask for the compressed
directories one at a time.

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
