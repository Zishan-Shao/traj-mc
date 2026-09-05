# Utilities

Standalone helpers retained from the original LLaDA experiment workspace:

- `generate.py` and `get_log_likelihood.py`: reference LLaDA inference helpers;
- `eval_benchmarks.py`: the generation-based benchmark runner that produces the
  LLaDA-Instruct low-rank table (`results/eval_stageB/`).  Besides the original
  MMLU / GSM8K / MATH-500 columns it runs SVAMP, AIME and Minerva Math, the
  zero-shot multiple-choice set (ARC-C, ARC-E, HellaSwag, PIQA) and
  HumanEval / MBPP / IFEval / BBH.  Every column streams its generations into
  `--gen_cache`, keyed by question, so a job that loses its GPU resumes;
- `oc_tasks/`: prompts, answer extraction, scoring and dataset access for those
  columns.  The code and instruction-following ones are ported from the
  OpenCompass configs the LLaDA repo's own evaluation script points at
  (`scripts/eval_llada_opencompass.sh`), so the sampler settings, few-shot
  prompts and judges match the published LLaDA-8B-Instruct protocol; the
  IFEval judge under `oc_tasks/ifeval/` is vendored verbatim.  `data.py` reads
  every dataset as the repo's own parquet/jsonl file, because the pinned
  `datasets` 2.16.1 raises on several of these repos before touching any data;
- protocol source of truth: the LLaDA repo's `evaluation/EVAL.md` carries a
  per-benchmark table of the published LLaDA-8B-Instruct settings (gen_length,
  block_length and the two eos flags), and `opencompass/examples/llada_instruct_*`
  names the OpenCompass dataset config each column uses. `BENCH_CONFIG` is a
  transcription of that table -- the columns genuinely differ (MMLU and
  HellaSwag generate 3 tokens, ARC-C generates 512), so do not harmonise them.
  LLaDA-8B-Instruct is evaluated by conditional **generation only**; the
  likelihood path in `trajmc/common.py`'s `LLADA_TASKS` is the Base model's.
  ARC-E and PIQA have no published Instruct config and inherit their nearest
  sibling (ARC-C and HellaSwag respectively);
- `collect_sink_results.py`: aggregate Sink-Aware evaluation output;
- `lora_finetune.py`: the earlier LoRA-on-compressed-model experiment.

The supported Traj-MC evaluation path is `python -m eval.run` (or
`trajmc-evaluate`). These utilities are kept for reproducibility and use only
repository-relative defaults. Generated data and weights belong under
`results/`, which is ignored by Git.

`eval_benchmarks.py` executes model-written Python for HumanEval and MBPP.
Each candidate program runs in its own subprocess under an address-space and
CPU-time limit with an empty environment (`utils/oc_tasks/sandbox.py`); that is
containment against a runaway generation, not against hostile code.

Datasets are pulled by file name rather than through `load_dataset`, because
the pinned `datasets==2.16.1` cannot read newer Hub metadata. Warm the cache on
a node with network access first:

```bash
python utils/eval_benchmarks.py --download_only
```
