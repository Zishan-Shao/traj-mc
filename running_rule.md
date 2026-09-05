# Preserving Mathematical Reasoning in Compressed Diffusion Language Models via Trajectory-Aware Low-Rank Approximation

Probably targets：https://mathai-2026.github.io/

地点确认是 **Atlanta**，deadline **September 25, 2026 AoE**，正文 **4 pages**，non-archival，accepted papers 全部现场 poster

## 1. Motivation: A Known Calibration Mismatch, but an Unresolved Compression Objective

dLLM compression 中存在一个已有且直观的问题：

$$
\text{clean calibration states} \neq \text{iterative generation states}.
$$

现有 activation-aware low-rank methods 通常沿用 conventional LLM setting，在 clean、fully visible activations 上定义 approximation quality：

$$
R_{\rm clean}(\widetilde W) = \mathbb E_{h\sim P_{\rm clean}} \left[ \|(W-\widetilde W)h\|_2^2 \right].
$$

而 dLLM inference 实际会沿 iterative denoising trajectory 访问一系列 timestep-dependent、partially masked states：

$$
h(x,t,m).
$$

**这一 mismatch 本身并不是我们的新发现。**

真正尚未被充分回答的是两个问题：

> **First, what low-rank approximation objective should correspond to this iterative generation process?**
> 

以及：

> **Second, how much does the choice of calibration distribution actually matter for downstream reasoning capability?**
> 

现有方法通常把 calibration state selection 当作数据构造或 heuristic choice，但没有系统回答：

$$
\boxed{ \text{Which population objective is being approximated when calibration spans the generation trajectory?} }
$$

更不知道这种 local approximation choice 是否只是影响 reconstruction metric，还是会进一步决定 mathematical reasoning 在 compression 后保留多少。

因此我们的核心 thesis 是：

$$
\boxed{ \textbf{A known calibration mismatch corresponds to different notions of low-rank optimality, and this distinction can have measurable consequences for mathematical reasoning.} }
$$

由此引出三个 RQ：

- **RQ1:** How should low-rank optimality be defined over an iterative dLLM trajectory?
- **RQ2:** Can the resulting trajectory-level optimum be approximated efficiently?
- **RQ3:** How much does the choice of low-rank objective affect mathematical reasoning under compression?

---

# 2. RQ1: How Should Low-Rank Optimality Be Defined Along a Generation Trajectory?

clean-state calibration 实际对应一个特定 population objective：

$$
R_{\rm clean}(\widetilde W).
$$

对于 dLLM，我们将这一 formulation 扩展到 generation trajectory，定义：

$$
\boxed{ R_{\rm traj}(\widetilde W) = \mathbb E_{x,t,m} \left[ \|(W-\widetilde W)h(x,t,m)\|_2^2 \right]. }
$$

其对应的 trajectory-integrated covariance 为：

$$
\boxed{ \Sigma_{\rm traj} = \mathbb E_{x,t,m} \left[ h(x,t,m)h(x,t,m)^\top \right]. }
$$

因此：

$$
R_{\rm traj}(\widetilde W) = \operatorname{Tr} \left[ (W-\widetilde W) \Sigma_{\rm traj} (W-\widetilde W)^\top \right].
$$

若：

$$
\Sigma_{\rm traj}=LL^\top,
$$

则 rank-`k` trajectory-aware approximation 为：

$$
\min_{\operatorname{rank}(\widetilde W)\le k} \|(W-\widetilde W)L\|_F^2,
$$

对应：

$$
\boxed{ W_{{\rm traj},k}^{\star} = (WL)_kL^\dagger. }
$$

这里的贡献不在于提出新的 SVD solver，而在于明确：

$$
\boxed{ R_{\rm clean} \text{ and } R_{\rm traj} \text{ define different population notions of low-rank optimality.} }
$$

clean-state calibration 可以被视为这一 trajectory formulation 的特殊情况，例如：

$$
\pi_{\rm clean}(t)=\delta(t=0).
$$

因此 RQ1 回答的是：

> **A calibration distribution is not merely a source of activations; it defines the geometry under which a low-rank approximation is optimal.**
> 

### 对应实验

这一节重点验证这种 distinction 是否真实而非纯形式上的：

- clean-state 与 trajectory-state activation statistics 是否显著不同；
- 两种 objective 是否产生不同的 low-rank solutions；
- clean-optimal 与 trajectory-optimal solutions 在对应 state distributions 上是否表现出明确的 reconstruction trade-off；
- 这种差异如何随 timestep、mask ratio、layer 和 compression severity 变化。

这一部分建立：

$$
\boxed{ \text{calibration distribution} \rightarrow \text{different approximation geometry}. }
$$

---

# 3. RQ2: Can We Efficiently Approximate the Trajectory-Level Optimum?

完整计算：

$$
\Sigma_{\rm traj} = \mathbb E_{x,t,m}[hh^\top]
$$

需要覆盖大量 examples、timesteps 和 state realizations，因此直接 exhaustive calibration 成本较高。

但 trajectory covariance 本身是一个 expectation。

因此我们采样：

$$
(x_i,t_i,m_i) \sim \mathcal D\times\pi(t)\times q(m\mid t),
$$

并构造 empirical covariance：

$$
\boxed{ \widehat\Sigma_N = \frac{1}{N} \sum_{i=1}^{N} h_ih_i^\top. }
$$

相应地：

$$
\boxed{ \widehat W_{N,k}^{\star} = (W\widehat L_N)_k\widehat L_N^\dagger, \qquad \widehat\Sigma_N = \widehat L_N\widehat L_N^\top. }
$$

因此 trajectory-state sampling 的意义不是：

> random masking happens to improve calibration.
> 

而是：

$$
\boxed{ \textbf{a Monte Carlo approximation to a explicitly defined trajectory-level population objective.} }
$$

在标准条件下：

$$
\widehat\Sigma_N \rightarrow \Sigma_{\rm traj},
$$

并进一步：

$$
\widehat W_{N,k}^{\star} \rightarrow W_{{\rm traj},k}^{\star}.
$$

从而形成：

$$
\boxed{ \text{population trajectory objective} \rightarrow \text{sampled covariance} \rightarrow \text{empirical optimum} \rightarrow \text{trajectory optimum}. }
$$

### 对应实验

RQ2 主要回答：

1. **Sample efficiency**
    
    $$
    N\uparrow \Rightarrow \widehat\Sigma_N \text{ and reconstruction quality stabilize}.
    $$
    
2. **Practical approximation**
    
    少量 sampled trajectory states 是否已经能够逼近更昂贵的 dense / exhaustive trajectory calibration。
    

因此这一节真正的 method claim 是：

> **The trajectory-level objective is not only well defined, but can also be approximated efficiently in practice.**
> 

---

# 4. RQ3: How Much Does This Choice Matter for Mathematical Reasoning?

前两个 RQ 建立的是：

$$
\text{different calibration distributions} \rightarrow \text{different local low-rank optima}.
$$

但一个更重要的问题仍然没有回答：

> **Does this local distinction actually matter for model behavior?**
> 

尤其对于 Math-AI，我们关注：

> **How much mathematical reasoning capability is lost simply because compression is optimized under a different state distribution?**
> 

在 matched backbone、rank / keep ratio、compression budget 和 calibration budget 下，我们比较：

$$
W_{\rm clean,k}^{\star} \qquad\text{vs.}\qquad W_{\rm traj,k}^{\star},
$$

并在 GSM8K、MATH-500 等 mathematical reasoning benchmarks 上进行系统评价。

如果 trajectory-aware compression 稳定优于 clean-state calibration，那么核心 finding 不是：

> random-`t` gives higher accuracy.
> 

而是：

$$
\boxed{ \textbf{The local state distribution used to define low-rank fidelity can materially determine how much mathematical reasoning survives compression.} }
$$

这一节需要重点**量化这种 behavioral consequence**：

- reasoning degradation under clean-state compression；
- trajectory-aware approximation 能恢复多少 compression-induced reasoning loss；
- effect 是否跨 models、ranks / keep ratios 和 reasoning benchmarks 稳定；
- reasoning improvement 是否与 trajectory reconstruction improvement 一致。

如果结果支持 reasoning 上的 effect 明显强于 general tasks，还可以进一步得到：

$$
\boxed{ \textbf{Mathematical reasoning is particularly sensitive to approximation errors accumulated over iterative generation states.} }
$$

否则保持更稳健的结论：

$$
\boxed{ \text{trajectory fidelity} \rightarrow \text{stronger mathematical reasoning preservation}. }
$$

这一节是全文的 empirical climax，也是 Math-AI relevance 最核心的部分。

---

# 5. Discussion: From a Known Mismatch to a Quantified Reasoning Consequence

全文最终回答的不是：

> Do clean calibration states differ from iterative states?
> 

这个问题本身早已具有直观基础。

我们真正推进的是：

$$
\boxed{ \text{known state mismatch} \rightarrow \text{explicit population objective} \rightarrow \text{principled estimator} \rightarrow \text{quantified reasoning consequence}. }
$$

因此 trajectory-aware compression 的意义也不只是一个 calibration trick。

它揭示了一个更 general 的 principle：

$$
\boxed{ \textbf{Compression fidelity is defined relative to the states over which model computation unfolds.} }
$$

而在 dLLMs 中，这个看似局部的 choice 会进一步传播到 mathematical reasoning：

$$
\boxed{ \text{state distribution} \rightarrow \text{low-rank geometry} \rightarrow \text{reasoning preservation}. }
$$

---

### 三条 contribution

> **First**, we formalize a known calibration mismatch in dLLM compression as a distinction between clean-state and trajectory-integrated notions of low-rank optimality, defining an explicit population objective over iterative generation states.
> 

> **Second**, we derive the corresponding fixed-rank activation-weighted SVD solution and develop a Monte Carlo trajectory-state estimator that efficiently approximates the population objective, with consistency under standard conditions.
> 

> **Third**, we quantify the behavioral consequence of this distinction and show that trajectory-aware low-rank approximation substantially better preserves mathematical reasoning under matched compression budgets, connecting local approximation geometry to global reasoning capability.
> 

最终整篇 paper 的 narrative 就是：

$$
\boxed{ \underbrace{\text{Known Problem}}_{\text{calibration mismatch}} \rightarrow \underbrace{\text{RQ1}}_{\text{formalize what it means}} \rightarrow \underbrace{\text{RQ2}}_{\text{solve it efficiently}} \rightarrow \underbrace{\text{RQ3}}_{\text{quantify why it matters for math}} }
$$

## Traj-MC: Adapting General Compression Methods to Discrete Diffusion Language Models via Monte Carlo Trajectory Calibration

## 1. Motivation

Traj-MC is a **drop-in calibration sampler for discrete diffusion language model compression**. In simple words it collects correct, distribution-matched samples in discrete dLLM in monte-carlo style. **Calibration is what the compressor does with the data; sampling is how we choose the data. Traj-MC only changes the latter.** Existing LLM compressors already define suitable calibration objectives,

$$
\mathcal L_C(\theta;P) =
\mathbb E_{z\sim P}
[\ell_C(\theta;z)],
$$

but typically estimate them using clean-text samples from $P_{\mathrm{clean}}$. A dLLM, however, repeatedly evaluates the same model on partially masked states induced by its deployed sampler.

Traj-MC therefore leaves the **objective and solver unchanged** and constructs a sampler-aligned calibration distribution

$$
P_{\mathrm{Traj}}(x,s,m)
=
P_{\mathrm{calib}}(x)\,
\pi(s)\,
P(m\mid s,g(x)),
$$

where $x$ comes from ordinary calibration text, $s$ is an actual sampler state, and $m$ follows the corresponding MASK cardinality and conditional geometry.

> **Traj-MC samples calibration states from the distribution induced by the deployed dLLM sampler.**
> 

Conceptually,

$$
\boxed{
\text{Sampler}
\rightarrow
\text{Traj-MC Samples}
\rightarrow
\text{Existing Compressor}
\rightarrow
\widehat W
}
$$

The **compression objective, solver, and inference graph remain unchanged**. We evaluate this sampling principle across established quantization, pruning, and low-rank compressors.

## 2. Related Work & Positioning

Recent dLLM compression methods have already introduced timestep- or mask-aware calibration to reduce the mismatch between clean text and iterative generation states. **Quant-dLLM** introduces Masked Calibration Simulation (MCS) to simulate timestep-dependent masking, while **DLLMQuant** uses Temporal-Mask Adaptive Sampling (TMAS) to cover different timestep and mask-ratio regimes. More recent work such as **FAIR-Calib** further reweights fragile deployment states according to frontier instability.

Traj-MC focuses specifically on the **sampling problem underlying calibration**. Rather than introducing another compressor-specific objective or solver, it formulates calibration samples as Monte Carlo samples from the sampler-induced deployment distribution.

Its intended novelty is therefore:

> **a compressor-agnostic deployment-distribution sampler for estimating existing compression objectives. Therefore LLM compressor can become competitive on discrete dLLM.**
> 

### Success Criteria

Traj-MC succeeds if we establish three points:

1. **Estimation:** Traj-MC consistently estimates the same compression objective under $P_{\mathrm{deploy}}$, with predictable finite-sample convergence.
2. **Generality:** the same sampling procedure can be plugged into multiple compression families without changing their objectives, solvers, or inference graphs.
3. **Practical relevance:** deployment-aligned sampling produces reproducible improvements over clean calibration and competitive diffusion-aware sampling baselines across multiple dLLMs and compressors.

The key novelty gate is:

> Traj-MC must be demonstrated as **a general deployment-distribution calibration framework**, not merely a more careful implementation of timestep/mask-aware PTQ calibration.
> 

---

## 3. Traj-MC

Traj-MC constructs calibration states from

$$
P_{\mathrm{Traj}}(x,s,m) =
P_{\mathrm{calib}}(x)
\pi(s)
P(m\mid s,g(x)),
$$

where $x$ is calibration text, $s$ is a deployed sampler state, and $m$ is a MASK realization consistent with that state and its conditional geometry. 

Using stratified Monte Carlo sampling, Traj-MC draws

$$
z_{s,i}=(x_{s,i},s,m_{s,i}),
\qquad i=1,\ldots,n_s,
$$

for each sampler state $s$. Under equal scheduler stratification,

$$
\pi_s=\frac{1}{T},
$$

so every deployed forward-call state receives equal calibration coverage. The resulting calibration set $\mathcal D_{\mathrm{Traj}} = {z_{s,i}}_{s,i}$ is passed directly to the existing compressor. If the compressor originally uses

$$
\mathcal L_C(\theta;P) =
\mathbb E_{z\sim P}
[\ell_C(\theta;z)],
$$

then it applies the **same** $\ell_C$ and the **same solver** to $\mathcal D_{\mathrm{Traj}}$; Traj-MC changes only the distribution from which its calibration samples are constructed.

### **What “Deployment Distribution” Means:**

Traj-MC currently does **not** collect full on-policy reverse rollouts from the dense model. Instead, it matches the deployment process in the dimensions directly specified by the sampler:

- forward-call schedule;
- remaining-MASK cardinality;
- visible-prefix geometry;
- generation-suffix geometry;
- distribution over MASK positions.

Semantic content is obtained from ordinary calibration text.

Operationally, the current method therefore constructs

$$
P_{\mathrm{Traj}} = 
P_{\mathrm{text}}

\times

P_{\mathrm{sampler\ state}\mid\mathrm{text}},
$$

where the sampler-state component matches deployed generation geometry. This is the distribution over which Traj-MC performs Monte Carlo integration.

### Terminology boundary

In presentation, we may say:

> **sampler-induced deployment distribution**
> 

but we must not imply that we explicitly collect on-policy reverse trajectories.

A precise sentence is:

> Traj-MC matches the sampler-induced state distribution while using ordinary calibration text to approximate its semantic marginal.
> 

The reverse-state alignment experiment will measure how closely this synthetic deployment marginal approximates true reverse-state statistics.

---

### Traj-MC Algorithm

Traj-MC changes only the **calibration distribution**. The compressor itself remains unchanged.

#### Algorithm 1: Traj-MC Calibration

**Input:** calibration corpus $\mathcal D$, deployed sampler $S$, calibration budget $N$, compressor $C$

**Output:** compressed model $\widehat W$

```
1. Extract the sampler states {s1, ..., sT} visited during generation.

2. For i = 1, ..., N:
      a. Sample a sampler state s.
      b. Sample a calibration sequence x ~ D.
      c. Keep the conditioning prefix visible.
      d. Mask the generation suffix according to state s.
      e. Add the resulting state zi to the calibration set DTraj.

3. Run the original compressor:
      W_hat = C(DTraj)

4. Return W_hat.
```

Equivalently, Traj-MC constructs $z_i \sim P_{\mathrm{deploy}}$ and lets the existing compressor estimate whatever calibration statistics it already requires:

$$
\widehat{\mu}_C = 
\frac{1}{N}
\sum_{i=1}^{N}
\phi_C(z_i).
$$

Thus, $C(\mathcal D_{\mathrm{clean}}) \longrightarrow C(\mathcal D_{\mathrm{TrajMC}})$ with **no change to the compressor or inference graph**.

---

## 4. Theory: From Trajectories to Calibration Occupancy

The theoretical contribution of Traj-MC is to formalize **what information from a deployment trajectory is actually needed for compression calibration**, and how this information can be approximated without running full reverse rollouts.

### 4.1 From trajectory paths to state occupancy

A full reverse trajectory is $\tau=(z_1,\ldots,z_T),$ with deployment path distribution

$$
\mathbb P_{\mathrm{deploy}}(d\tau) = 
Q_1^{\mathrm{deploy}}(dz_1)
\prod_{s=1}^{T-1}
K_s^{\theta,S}(dz_{s+1}\mid z_s).
$$

Traj-MC does not model this full joint path law. Instead, it considers the per-step marginals

$$
Q_s^{\mathrm{deploy}}(dz)
\Pr(z_s\in dz),
$$

which induce the trajectory occupancy measure

$$
\boxed{\nu_{\mathrm{deploy}}(ds,dz) =
\sum_{s=1}^{T}
\pi_s
\delta_s(ds)
Q_s^{\mathrm{deploy}}(dz)
}.
$$

For calibration statistics that depend on individual forward states,

$$
F(\tau) = 
\sum_{s=1}^{T}
\pi_s\phi(z_s),
$$

their deployment expectation depends only on the occupancy measure:

$$
\mathbb E_{\tau\sim\mathbb P_{\mathrm{deploy}}}[F(\tau)] =
\mathbb E_{(s,z)\sim\nu_{\mathrm{deploy}}}[\phi(z)].
$$

Thus, **state-local calibration does not require recovering the full trajectory path distribution; matching its state occupancy measure is sufficient.**

---

### 4.2 Traj-MC as a surrogate occupancy measure

Directly sampling $\nu_{\mathrm{deploy}}$ requires reverse rollouts. Traj-MC instead constructs a tractable surrogate using ordinary calibration text:

$$
x\sim P_{\mathrm{calib}},
\qquad
s\sim\pi,
\qquad
m\sim Q_s^{\mathrm{Traj}}(\cdot\mid g(x)),
$$

$z=\Psi(x,m,g(x)).$ This induces

$$
\boxed{\nu_{\mathrm{Traj}}(ds,dz) =
\sum_{s=1}^{T}
\pi_s\delta_s(ds)
\int
P_{\mathrm{calib}}(dx)
Q_s^{\mathrm{Traj}}(dm\mid g(x))
\delta_{\Psi(x,m,g(x))}(dz)
}.
$$

For a locked sampler configuration, Traj-MC exactly matches the forward-call schedule and remaining-MASK cardinality, while using randomized mask placement and ordinary calibration text as tractable proxies for the corresponding on-policy states.

---

### 4.3 Finite-sample error and surrogate alignment

For any compressor-relevant state statistic $\phi_C$, define

$$
\mu_C^{\mathrm{deploy}} =
\mathbb E_{\nu_{\mathrm{deploy}}}[\phi_C(z)],\qquad\mu_C^{\mathrm{Traj}} =
\mathbb E_{\nu_{\mathrm{Traj}}}[\phi_C(z)].
$$

Given finite Traj-MC samples, let $\widehat\mu_C$ denote the corresponding Monte Carlo estimate. Then

$$
\boxed{|\widehat{\mu}_C - \mu_C^{\mathrm{deploy}}|\le\underbrace{  |\widehat{\mu}_C - \mu_C^{\mathrm{Traj}}|}_{\text{finite-sample MC error}}+\underbrace{  |\mu_C^{\mathrm{Traj}} - \mu_C^{\mathrm{deploy}}|}_{\text{surrogate alignment gap}}}
$$

This separates the two sources of error in Traj-MC:

1. **Monte Carlo error** decreases as calibration coverage increases. Under standard finite-variance assumptions, it exhibits the usual $O_p(N^{-1/2})$ scaling.
2. **Surrogate alignment gap** does not vanish with more Traj-MC samples and must instead be validated by comparing synthetic and real reverse states in compression-relevant statistics.

The resulting theoretical view is:

$$
\boxed{
\text{deployment trajectory}
\rightarrow
\text{occupancy measure}
\rightarrow
\text{Traj-MC surrogate}
\rightarrow
\text{finite-sample estimate}
}
$$

Traj-MC therefore does not attempt to reproduce the full reverse trajectory. It approximates the **trajectory-induced state occupancy needed by state-local compression calibration**.

---

## 5. Claims

The paper should have a strict claim hierarchy.

### Claims we have made:

#### Claim 1 — Distribution mismatch

Clean-endpoint calibration does not represent the states repeatedly encountered during dLLM iterative generation.

#### Claim 2 — Formulation

Compression calibration can be formulated as estimation of compressor-specific quantities under a sampler-induced deployment distribution.

#### Claim 3 — Method

Traj-MC provides a simple Monte Carlo estimator of these deployment calibration quantities using sampler-aligned masked states.

#### Claim 4 — Generality

The same Traj-MC calibration interface can be used by multiple existing compression families without modifying their solver or inference graph.

#### Claim 5 — Statistical behavior

Traj-MC calibration statistics converge with increasing Monte Carlo coverage, and finite calibration coverage explains part of the observed run-to-run variability.

#### Claim 6 — Empirical benefit

When clean/deployment mismatch is consequential, replacing clean calibration with Traj-MC can substantially preserve compressed dLLM generation quality at the same compression budget.

Do not claim that:

- Traj-MC collects true on-policy reverse trajectories;
- Traj-MC directly optimizes future error propagation;
- scheduler stratification always beats Random-t;
- Traj-MC improves every model and every compressor;
- Traj-MC requires a new compression solver;
- SVD is the central algorithmic contribution;
- (N^{-1/2}) convergence alone constitutes the theoretical novelty;
- a larger number of denoising steps necessarily increases Traj-MC's benefit.

Negative results should remain visible rather than being hidden through benchmark selection.

## 6. Experiment Matrix

Benchmark Setting：

$$
\boxed{
\text{General Understanding}
+
\text{Reasoning}
+
\text{Instruction Following}
+
\text{Coding}
}
$$

| Capability | Core benchmarks |
| --- | --- |
| General Understanding | MMLU、ARC-C、HellaSwag、PIQA、WinoGrande |
| Reasoning | GSM8K、MATH-500、BBH、GPQA |
| Instruction Following | IFEval |
| Coding | HumanEval+、MBPP+ |

Model Variants

| Model variant | General | Reasoning | Instruction | Coding |
| --- | --- | --- | --- | --- |
| Base | ✓ | ✓ | — (do not do instruction with base model) | ✓ |
| Instruct | optional | ✓ | ✓ | ✓ |
|  |  |  |  |  |

### Main Experiment — LLaDA (Part 1)

**Cell format:** `☐ Clean: __ / ☐ Traj: __`

| Family | Method | Setting | Checkpoint | GSM8K | BBH | HumanEval | MBPP | MMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | BF16 | — | ☐ | 70.1 | 44.7 | 35.4 | 39.4 | 66.0 |
| **Quant** | **GPTQ** | **W4-G128** | ☐ **C / ☐ T | ☐ 68.3 / 70.1 | ☐ 44.9 ****/ 44.5 | ☐ 30.5 / 32.3 | ☐ 38.6 / 38.0 | ☐ 65.8 / 65.3 |
| **Quant** | **GPTQ** | **W3-G128** | ☐ C / ☐ T | ☐ 58.7 / 59.6 | ☐ 43.1 / 43.5 | ☐ 20.1 / 22.0 | ☐ **33.4 / 33.4** | ☐ 62.9 / 62.7 |
| Quant | AWQ | W4-G128 | ☐ C / ☐ T | ☐ 68.2 / 69.3 | ☐ 44.6 / 44.7 | ☐ 29.3 ****/ 28.7 | ☐ 37.8 / 38.6 | ☐ 64.9 / 65.0 |
| Quant | AWQ | W3-G128 | ☐ C / ☐ T | ☐ **55.5** / 54.1 | ☐ 39.6 / 39.5 | ☐ **23.2** / 19.5 | ☐ **32.8** / 30.4 | ☐ **60.8** / 60.4 |
| Low-rank | SVD-LLM | 20% reduction | ☐ C / ☐ T | ☐ 19.3 / 36.3 | ☐ **40.0** / 39.1 | ☐ 2.4 / 2.4 | ☐ 1.0 / 0.0 | ☐ 58.0 / 56.8 |
| Low-rank | SVD-LLM | 40% reduction | ☐ C / ☐ T | ☐ 0.0 / **2.4** | ☐ 14.7 / **20.2** | ☐ 0.0/0.0 | ☐0.0/0.0 | ☐ 40.7 / **42.9** |
| Low-rank | ASVD | 20% reduction | ☐ C / ☐ T | ☐ **34.3** / 33.7 | ☐ **40.4** / 39.9 | ☐ 10.4 / 11.6 | ☐ 13.4 / 13.8 | ☐ 56.7 / 56.7 |
| Pruning
(unstructured) | SparseGPT | 50% sparsity | ☐ C / ☐ T | ☐ 54.7 / 55.2 | ☐ 41.7 / **42.8** | ☐ 11.0 / 14.0 | ☐ 24.0 / **26.0** | ☐ 60.9 / 61.1 |
| Pruning
(unstructured) | SparseGPT | 70% sparsity | ☐ C / ☐ T | ☐ 3.6 / 4.1 | ☐ **34.2** / 32.0 | ☐ **0.6 / 0.6** | ☐ 0.2 / **0.8** | ☐ 43.3 / 41.9 |
| Pruning
(unstructured) | Wanda | 50% sparsity | ☐ C / ☐ T | ☐ 55.5 / **55.9** | ☐ 40.7 / **41.0** | ☐ 14.6 / 15.2 | ☐ **23.6** / 23.0 | ☐ 61.3 / 61.5 |
| dLLM pruning | Sink-Aware Pruning
https://arxiv.org/pdf/2602.17664 | 50% sparsity | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |
| dLLM quant | quant-dllm (ICLR 2026)
https://openreview.net/forum?id=HD7tuVakmR | W4
**(这篇文章应该是2bit的，这个数据都是做的2bit的）** | ☐ | ☐ 37.5 | ☐ 37.0 | ☐ 10.4 | ☐ 18.6 | ☐ 54.9 |

### LLaDA Instruct (new version)

**Cell format:** `☐ Clean: __ / ☐ Traj: __`

| Family | Method | Setting | Checkpoint | GSM8K | **MATH-500** | HumanEval | MBPP | MMLU | IFEval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | BF16 | — | ☐ | **80.21** | **30.20** | __ | __ | **64.30** | __ |
| **Quant** | **GPTQ** | **W4-G128** | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| **Quant** | **GPTQ** | **W3-G128** | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Quant | AWQ | W4-G128 | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Quant | AWQ | W3-G128 | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | SVD-LLM | 20% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | SVD-LLM | 40% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | ASVD | 20% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning
(unstructured) | SparseGPT | 50% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning
(unstructured) | SparseGPT | 70% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning
(unstructured) | Wanda | 50% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| dLLM pruning | Sink-Aware Pruning
https://arxiv.org/pdf/2602.17664 | 50% sparsity | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |
| dLLM quant | quant-dllm (ICLR 2026)
https://openreview.net/forum?id=HD7tuVakmR | W4 | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |
|  |  |  |  |  |  |  |  |  |  |

| Family | Method | Setting | **Calibration** | GSM8K | **MATH-500** | MMLU |
| --- | --- | --- | --- | --- | --- | --- |
| Dense | BF16 | — | ☐ | **80.21** | **30.20** | **64.30** |
| Low-rank | Weight SVD | 20% red. | 无 | 12.28 | 4.8 | 39.66 |
| Low-rank | Clean-SVD | 20% red. | C4, t=0 | 42.23 | 9.00 | 57.19 |
| Low-rank | **Traj-SVD** | 20% red. | C4, t~U (0,1) | 56.71 | 11.00 | 57.26 |
| Low-rank | MCS-SVD | 20% red. | C4, grid-t + 25% 前缀 | 58.76 | 10.40 | 57.37 |
| Low-rank | *Actual-State* | 20% red. | *GSM8K rollout* | 75.36 | 23.80 | 53.69 |
| Low-rank | Weight SVD | 40% red. | 无 | 0.0 | 0.20 | 0.04 |
| Low-rank | Clean-SVD | 40% red. | C4, t=0 | 4.17 | 2.00 | 45.77 |
| Low-rank | **Traj-SVD**  | 40% red. | C4, t~U (0,1) | 11.22 | 2.40 | 48.49 |
| Low-rank | *Actual-State* | 40% red. | *GSM8K rollout* | 67.10 | 11.20 | 39.00 |

### Main Experiment — Dream (Part 2)

**Cell format:** `☐ Clean: __ / ☐ Traj: __`

| Family | Method | Setting | Checkpoint | GSM8K | BBH | HumanEval | MBPP | MMLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | BF16 | — | ☐ | __ | __ | __ | __ | __ |
| **Quant** | **GPTQ** | **W4-G128** | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| **Quant** | **GPTQ** | **W3-G128** | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Quant | AWQ | W4-G128 | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Quant | AWQ | W3-G128 | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | SVD-LLM | 20% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | SVD-LLM | 40% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | ASVD | 20% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning (unstructured) | SparseGPT | 50% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning (unstructured) | SparseGPT | 70% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning (unstructured) | Wanda | 50% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| dLLM pruning | Sink-Aware Pruning
https://arxiv.org/pdf/2602.17664 | 50% sparsity | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |
| dLLM quant | quant-dllm (ICLR 2026)
https://openreview.net/forum?id=HD7tuVakmR | W4 | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |

### Dream Instruct (Part 2)

**Cell format:** `☐ Clean: __ / ☐ Traj: __`

| Family | Method | Setting | Checkpoint | GSM8K | BBH | HumanEval | MBPP | MMLU | IFEval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | BF16 | — | ☐ | __ | __ | __ | __ | __ | __ |
| **Quant** | **GPTQ** | **W4-G128** | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| **Quant** | **GPTQ** | **W3-G128** | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Quant | AWQ | W4-G128 | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Quant | AWQ | W3-G128 | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | SVD-LLM | 20% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | SVD-LLM | 40% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Low-rank | ASVD | 20% reduction | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning (unstructured) | SparseGPT | 50% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning (unstructured) | SparseGPT | 70% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| Pruning (unstructured) | Wanda | 50% sparsity | ☐ C / ☐ T | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ | ☐ __ / __ |
| dLLM pruning | Sink-Aware Pruning
https://arxiv.org/pdf/2602.17664 | 50% sparsity | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |
| dLLM quant | quant-dllm (ICLR 2026)
https://openreview.net/forum?id=HD7tuVakmR | W4 | ☐ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ | ☐ __ |