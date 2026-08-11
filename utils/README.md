# Utilities

Standalone helpers retained from the original LLaDA experiment workspace:

- `generate.py` and `get_log_likelihood.py`: reference LLaDA inference helpers;
- `eval_benchmarks.py`: the earlier generation-based benchmark runner;
- `collect_sink_results.py`: aggregate Sink-Aware evaluation output;
- `lora_finetune.py`: the earlier LoRA-on-compressed-model experiment.

The supported Traj-MC evaluation path is `python -m eval.run` (or
`trajmc-evaluate`). These utilities are kept for reproducibility and use only
repository-relative defaults. Generated data and weights belong under
`results/`, which is ignored by Git.
