# Scripts

- `calibrate_pair.sh [llada|dream]`: build paired clean-`t0` and random-`t`
  calibration tensors with identical clean windows.
- `compress_arm.sh BACKEND ARM CALIBRATION_PT`: compress one arm.
- `evaluate.sh BACKEND TASK ARM [WEIGHTS_DIR]`: launch the supported lm-eval
  adapter.
- `run_random_t.sh [llada|dream]`: run paired calibration and compression.
- `run_stage_a.sh [BACKEND]`: Traj-SVD Stage A end to end -- five matched
  arms (`weight_svd`, `clean`, `mcs`, `traj`, `actual`), the RQ1 subspace
  distance, and the RQ2 generation-state reconstruction error. See
  `docs/TRAJ_SVD.md`.
- `lib.sh`: shared helpers that reconstruct the artifact names
  `trajmc-calibrate` writes (the scheme tag for `grid_t_prefix`/`rollout` and
  the seed tag for a split window/sampling seed).
- `run_table_d.sh`: fill the HumanEval / MBPP / IFEval columns of the
  LLaDA-Instruct low-rank table on the checkpoints that already carry its
  GSM8K / MATH-500 / MMLU numbers -- one Slurm job per (existing arm x
  benchmark), no new arms. BBH is implemented but opt-in (`BM=bbh`). Sampler
  settings come from `utils/eval_benchmarks.py`, which follows the LLaDA
  repo's own OpenCompass config per benchmark.
- `run_eval_workers.sh`: fill the remaining columns of the LLaDA-Instruct
  low-rank table -- SVAMP, ARC-C, ARC-E, HellaSwag, PIQA (plus AIME and
  Minerva Math) -- from a shared queue worked by GPU workers that keep their
  card until the queue is empty (`slurm/eval_worker.sbatch`), rather than
  handing it back between cells. Reads `N_NODE5`, `N_NODE6`, `N_ANY`,
  `ANY_PARTITION`, `BM`, `CELLS`, and `DRY`.
- `run_sampling_ablation.sh [llada|dream]`: build and audit matched
  `random_t`, `grid_t`, `grid_t_prefix`, and real `rollout` calibrations. Set
  `STAGE=all` to compress all four after the audit.
- `slurm/`: portable submission templates without cluster-specific accounts,
  partitions, environments, or absolute paths.

Configuration is supplied through environment variables such as `MODEL_PATH`,
`NSAMPLES`, `SEED`, `SAMPLING_SEED`, `EVAL_SEED`, `RATIO`, `LAYER_TYPE`,
`XTX_BUDGET_GB`, `NUM_PROCESSES`, `PREFIX_RATIO`, `ROLLOUT_STEPS`, and `STAGE`.
`run_table_d.sh` additionally reads `PARTITION`, `BBH_PER_SUBTASK`, `BM`,
`CELLS`, and `DRY`.

`analysis/stage_b_table.py --benches wide` prints the assembled table (GSM8K,
MATH-500, MMLU, SVAMP, ARC-C, ARC-E, HellaSwag, PIQA), leaving unfinished
cells blank so it doubles as a progress sheet.

All outputs default to the ignored `results/` tree.
