# Baselines

Vendored source snapshots for the external comparison methods used around the
Traj-MC experiments:

- `quant_dllm/`: [Quant-dLLM](https://github.com/ZTA2785/Quant-dLLM), including
  its upstream Apache-2.0 license and notices;
- `sink_aware_pruning/`:
  [Sink-Aware Pruning](https://github.com/VILA-Lab/Sink-Aware-Pruning),
  including its upstream MIT license.

Nested Git metadata, figures, caches, checkpoints, datasets, and raw results
were intentionally omitted. Each directory remains a standalone upstream-style
code tree; run its entry points from inside that directory so its local imports
resolve exactly as upstream intended.

Traj-MC's own clean-`t0` comparison arm is implemented by
`trajmc.calibration --arm base`; it is not duplicated here.
