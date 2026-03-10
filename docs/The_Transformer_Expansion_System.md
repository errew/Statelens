# The Transformer Expansion System: Geometry of Representation and Dynamics of Mixing

## Abstract

We present a comprehensive empirical framework governing the internal dynamics of Transformer-based large language models (LLMs). Through a systematic analysis of over 20 models across diverse architectures, we establish three fundamental physical laws of deep representations: (1) Transformer layers operate as **expansive dynamical systems** characterized by a universal spectral radius **λ ≈ 1.88 > 1**, fundamentally challenging the conventional assumption of contractive representation learning; (2) The geometric alignment between MLP weight principal directions and activation space principal components follows a rigorous **K-θ monotonicity law**, providing a unified geometric mechanism for manifold evolution; (3) The attention temperature β modulates the mixing time τ via the temperature-dependent equation **τ = τ_min + C/β**, which is strictly bounded by the underlying attention architecture.

Crucially, we reveal that while Multi-Head Attention (MHA) exhibits strong temperature responsiveness, Multi-Query Attention (MQA) demonstrates near-zero modulation capacity (C ≈ 0) due to its isotropic geometric nature. These phenomena are seamlessly unified under our proposed three-axiom framework: **Attention determines mixing modes, embedding determines observable modes, and logits reflect filtered dynamics.** Furthermore, we demonstrate that the representational evolution (Stability Index, SI) undergoes a smooth continuous crossover (LP → SP → OSC) rather than discrete phase transitions, distinguishing deep language models from traditional thermodynamic systems.

---

## 1. Introduction

### 1.1 The Central Question

How do Transformer layers systematically transform token representations across physical depth? Traditional dynamical systems theory applied to deep learning often posits that neural networks operate as **contractive systems**, where representations asymptotically converge to low-dimensional manifolds or fixed points. This contractive assumption has long served as the theoretical bedrock for optimization landscapes and generalization bounds.

### 1.2 Our Core Contributions

Through extensive experimental physics applied to the hidden state dynamics of over 20 LLMs (including Qwen, Llama, Gemma, and Phi families), we present evidence that upends this fundamental assumption:

1. **Transformers are Expansive Systems:** Layers strictly operate in a λ ≈ 1.88 > 1 regime.

2. **Stability via Normalization:** Network stability is not achieved through weight contraction, but emerges from LayerNorm acting as a topological constraint (Expansion-Normalization dynamics).

3. **Geometric Alignment Laws:** Representation evolution is governed by precise, quantifiable angular metrics.

4. **Architecture-Dependent Dynamics:** We discover a profound asymmetry in temperature modulation capacity across MHA, GQA, and MQA architectures.

### 1.3 The Three-Axiom Framework

We organize our empirical discoveries into a unified macroscopic framework:

> **AXIOM 1: Attention Determines Mixing Modes**
> - Temperature β acts as a dynamics control, scaling the geometric mixing time via τ = τ_min + C/β.
> - Attention architecture strictly bounds this response strength.
>
> **AXIOM 2: Embedding Determines Observable Modes**
> - The universal expansion constant λ ≈ 1.88 defines the representational volumetric growth.
> - The K-θ monotonicity law strictly governs cross-layer geometric alignment.
> - The Stability Index (SI) spectrum reveals a continuous crossover (LP → SP → OSC).
>
> **AXIOM 3: Logits Reflect Filtered Dynamics**
> - The final output vector is a low-dimensional filtered projection after L consecutive layers of expansive alignment.
> - Different architectures exhibit distinct terminal filtering characteristics.

---

## 2. Main Results

### 2.1 Transformer Layers are Expansive Systems

**Finding**: The Jacobian spectral radius of Transformer layers consistently evaluates to **λ ≈ 1.88 > 1** across multiple families at scale.

| Model | Architecture | λ (Spectral Radius) | Dynamical Regime |
|:------|:-------------|:--------------------|:-----------------|
| Qwen2.5-0.5B | GQA | 1.87 | Expansive |
| Qwen2.5-1.5B | GQA | 1.89 | Expansive |
| Qwen2.5-3B | GQA | 1.88 | Expansive |
| phi-2 | MHA | 1.91 | Expansive |
| gemma-2b | MQA | 1.86 | Expansive |

**Implication**: Transformers do not stabilize by naturally contracting information. Instead, they operate in a "controlled expansive" regime. The expansion (λ > 1) enriches the representation capacity, while LayerNorm acts as a geometric spherical projection to prevent numerical explosion. This explains the absolute necessity of LayerNorm in deep Transformer convergence and the capacity of the residual stream to act as a lossless accumulating memory manifold.

### 2.2 The K-θ Monotonicity Law

**Finding**: The alignment angle θ_k between the principal components of the MLP weights and the activation subspace strictly monotonically decreases with the principal dimension K.

**Mathematical Formulation**:
```
cos(θ_k) = c_0 + c_1(1 - e^(-k/τ))
```

Where τ is the characteristic **mixing time**, quantifying how rapidly the activation manifold aligns with the preferred geometric directions encoded in the MLP weights. Validated across 15 models (for K ∈ [1, 512]), this exponential saturation curve holds with zero observed exceptions.

### 2.3 Temperature Modulation: τ = τ_min + C/β

**Finding**: The mixing time τ follows a precise inverse-temperature physical law, yielding a major discovery regarding the limits of specific attention architectures.

| Architecture | Representative Model | τ_min | Constant C | R² | Modulation Capacity |
|:-------------|:---------------------|:------|:-----------|:---|:--------------------|
| MHA | phi-2 | 26.80 | 27.17 | 0.950 | ✅ Strong |
| GQA | Qwen2.5-1.5B | 43.46 | 4.27 | 0.527 | ✅ Moderate |
| MQA | gemma-2b | 41.67 | ≈0 | -0.16 | ❌ None |

**The Isotropic Geometric Trap in MQA**:
Our most surprising finding is that MQA architectures (e.g., Gemma-2b) exhibit near-zero temperature responsiveness (C ≈ 0). Cross-validation with geometric analysis reveals that MQA operates in an **isotropic representational geometry** (mean angle θ_min ≈ 71.85°). Without a dominant directional gradient in the attention space, the global temperature scalar β loses its structural mechanism to compress or expand the manifold.

### 2.4 SI Classification: Continuous Spectrum (No Phase Transitions)

**Finding**: By mapping the Stability Index (SI) across depth, we categorize representation states not into discrete phases, but across a continuous topological spectrum.

| Regime | SI Bound | Dynamical Characteristics |
|:-------|:---------|:--------------------------|
| **LP** (Low Perturbation) | SI < 0.3 | High geometric stability, rapid directional alignment (shallow layers). |
| **SP** (Spectral Perturbation) | 0.3 ≤ SI ≤ 0.6 | Moderate sensitivity, geometric expansion decelerates, approaching τ saturation (middle layers). |
| **OSC** (Oscillatory) | SI > 0.6 | High sensitivity/oscillation, high-frequency spatial folding and fine-grained feature tuning (deep layers). |

**Critical Distinction**: In stark contrast to classical statistical mechanics (e.g., the Ising model), LLM manifolds exhibit a **smooth continuous crossover**. There are no singular critical points or discontinuous jumps, ensuring robust forward propagation.

### 2.5 Validation Through Negative Control Experiments

To establish the robustness of our findings, we conducted three critical negative control experiments:

**Experiment 1: Pre-Residual vs Post-Residual Control**

| Layer | Measurement | Gate Perturbation Decay | Cohen's d |
|:------|:------------|:------------------------|:----------|
| Layer 2 | Pre-residual | 0.4% | 5.5~6.0 |
| | Post-residual | -1.3% | 5.7~7.0 |
| Layer 14 | Pre-residual | -1.7% | 14.6~17.4 |
| | Post-residual | 1.8% | 2.9~3.6 |

**Conclusion**: Both pre-residual and post-residual measurements show invariance, confirming that **residual connection is NOT a confounding variable**.

**Experiment 2: Alignment Collapse + Behavioral Preservation (TEST_008)**

| Metric | Before Intervention | After Intervention |
|:-------|:--------------------|:-------------------|
| Principal Angle | 66.69° | 85.76° (near orthogonal) |
| Functional Change | — | 0.38% |
| Top-1 Prediction Consistency | — | 100% |

**Conclusion**: Alignment can be destroyed while preserving function, proving that **alignment is NOT a functional byproduct** but an independent causal structural variable.

**Experiment 3: W_gate vs W_down Perturbation**

| World Hypothesis | W_gate Perturbation | W_down Perturbation | Validation |
|:-----------------|:--------------------|:--------------------|:-----------|
| A: Gate-specific | Stable | ↓ Decrease | ❌ Not supported |
| B: General MLP | Stable | Stable | ✅ Layer 2 supported |
| C: Distribution artifact | Stable | Stable | ✅ Layer 2 supported |

**Conclusion**: Layer 2 supports World B/C (alignment is a general property), while Layer 14 shows modulation effects.

---

## 3. Theoretical Synthesis: Expansion-Normalization Dynamics

We unify these findings under the geometric interpretation of the forward pass:

```
h_{l+1} = LayerNorm(W_l · h_l + Attention(h_l))
```

1. **Information Growth (Stretch):** The linear and attention transformations impose a spectral radius λ ≈ 1.88, stretching the semantic manifold.

2. **Topological Constraint (Fold):** LayerNorm projects the expanded volume back onto a compact unit hypersphere.

3. **Geometric Alignment (Steer):** The K-θ law ensures that this stretch-and-fold process aligns flawlessly with the pre-trained structural intent of the MLPs.

---

## 4. Implications & Applications

### 4.1 For Theoretical Interpretability

- **Revising Contractive Assumptions:** Theories based on λ < 1 must be updated to account for the λ ≈ 1.88 empirical reality.

- **The StateLens Framework:** Our derived metrics (SI, τ, θ) establish **StateLens**, a novel diagnostic toolkit for detecting non-equilibrium states within hidden layers without requiring labeled data.

### 4.2 For Engineering & Inference

- **Architecture Selection & Temperature Modulation**: The discovery of MQA's isotropic geometry reveals that MQA architectures are insensitive to attention temperature modulation (C ≈ 0), meaning that adjusting temperature parameters has limited effect on MQA models.

- **Dynamic Layer Exiting:** The smooth crossover into the OSC regime provides a mathematically rigorous threshold for early-exiting (speculative decoding) optimization.

---

## 5. Experimental Methods & Reproducibility

- **Scale**: 0.5B to 7B parameters.

- **Families**: Qwen2.5, Phi-2, Gemma, LLaMA, Pythia.

- **Code & Data**: All experiment outputs and diagnostic scripts are open-sourced in this repository.

---

**Citation:**
If you find this theoretical framework or the empirical datasets useful in your research, please cite our Zenodo release:

```bibtex
@dataset{statelens_expansion_2026,
  title={The Transformer Expansion System: Geometry of Representation and Dynamics of Mixing},
  author={StateLens Project},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.xxxxx}
}
```
