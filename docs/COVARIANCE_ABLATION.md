# Timestep sampling and covariance estimation

## Question

Does iid `t ~ Uniform(0,1)` estimate the diffusion-marginal activation second
moment better than the stratified grid of released Quant-dLLM MCS?

Note which scheme is the published baseline: `grid_t_prefix` reproduces
`baselines/quant_dllm/utils/mcs.py::apply_mcs` byte for byte, prefix included
(`prefix_ratio=0.25` is that function's default). `grid_t` is the same grid
with the prefix switched off and corresponds to no published method; it exists
only to separate the timestep axis from the prefix axis.

For activation vector `h(x_t)`, the target is

```text
S = E_x E_t E_mask [h(x_t) h(x_t)^T].
```

Iid random-`t` is an unbiased Monte Carlo estimator of the continuous-time
expectation. The fixed endpoint grid is a finite quadrature rule: it can have
lower variance when the integrand is smooth in `t`, but can retain quadrature
bias when activation covariance is nonlinear in `t`. Neither outcome should be
assumed in advance.

## Matched four-way ablation

Keep backend, clean windows, compression settings, factor rank, and evaluation
protocol fixed. Change only:

1. `random_t`: iid timestep, full-window Bernoulli corruption;
2. `grid_t`: deterministic timestep grid, full-window Bernoulli corruption
   (axis-isolation arm, not a published method);
3. `grid_t_prefix`: the same grid with a fixed visible prefix -- this is the
   released Quant-dLLM MCS;
4. `rollout`: the same grid mapped to real reverse-sampler calls, with the same
   visible prefix.

The calibration audit fails unless all four artifacts contain byte-identical
`windows_pre`. LLaDA rollout states are selected by remaining MASK count, not
raw call index, so a non-uniform per-call transfer schedule still matches the
requested mask-ratio grid. When transfers are uniform and
`N == rollout_steps`, the rollout arm covers every native pre-forward sampler
call exactly once.

## Covariance protocol

Use `analysis.covariance_estimation` with:

- a large, independent reference artifact;
- at least five independently seeded `random_t` artifacts;
- independently seeded mask patterns for grid arms when estimating their mask
  variance;
- identical model, layers, coordinate seed, and sample cap.

Report per-layer and mean relative Frobenius, spectral, and trace error. The
diagnostic uses exact covariance on a deterministic activation-coordinate
subspace, making several 8B-model layers practical without claiming that the
subspace is the full covariance.

Use a fixed `--seed` for clean-window selection and vary only
`--sampling_seed`; this prevents data-sampling noise from being mistaken for a
timestep-sampling effect.

An iid-specific covariance advantage requires lower error across repeated
seeds, not a single favorable checkpoint. If grid wins, the correct conclusion
is that stratification is a better estimator at that budget. If only rollout
wins downstream, the gain comes from model-feedback states rather than
timestep coverage alone.
