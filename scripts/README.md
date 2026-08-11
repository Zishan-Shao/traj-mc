# Scripts

- `calibrate_pair.sh [llada|dream]`: build paired clean-`t0` and random-`t`
  calibration tensors with identical clean windows.
- `compress_arm.sh BACKEND ARM CALIBRATION_PT`: compress one arm.
- `evaluate.sh BACKEND TASK ARM [WEIGHTS_DIR]`: launch the supported lm-eval
  adapter.
- `run_random_t.sh [llada|dream]`: run paired calibration and compression.
- `slurm/`: portable submission templates without cluster-specific accounts,
  partitions, environments, or absolute paths.

Configuration is supplied through environment variables such as `MODEL_PATH`,
`NSAMPLES`, `SEED`, `RATIO`, `LAYER_TYPE`, `XTX_BUDGET_GB`, and
`NUM_PROCESSES`. All outputs default to the ignored `results/` tree.
