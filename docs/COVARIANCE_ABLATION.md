# Timestep sampling and covariance estimation

## Question

Does iid `t ~ Uniform(0,1)` estimate the diffusion-marginal activation second
moment better than the deterministic grid used by released Quant-dLLM MCS?

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
2. `grid_t`: deterministic timestep grid, full-window Bernoulli corruption;
3. `grid_t_prefix`: the same grid with a fixed visible prefix;
4. `rollout`: the same grid mapped to real reverse-sampler calls, with the same
   visible prefix.

The calibration audit fails unless all four artifacts contain byte-identical
`windows_pre`. For `N == rollout_steps`, the rollout arm covers every native
pre-forward sampler call exactly once.

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

An iid-specific covariance advantage requires lower error across repeated
seeds, not a single favorable checkpoint. If grid wins, the correct conclusion
is that stratification is a better estimator at that budget. If only rollout
wins downstream, the gain comes from model-feedback states rather than
timestep coverage alone.
