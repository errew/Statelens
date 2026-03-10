# StateLens: The Transformer Expansion System

**[中文版 (Chinese Version)](README_CN.md)**

> **Attention determines mixing modes, embedding determines observable modes, logits reflect filtered dynamics.**

## Overview

This repository contains the experimental data and analysis code for our comprehensive study of Transformer internal dynamics. We present evidence that Transformer layers operate as **expansive systems** (λ ≈ 1.88 > 1), challenging the conventional assumption of contractive representation learning.

## Key Findings

### 1. Transformer Layers are Expansive Systems
- Spectral radius λ ≈ 1.88 > 1 across 15+ models
- Stability emerges from LayerNorm, not weight contraction

### 2. K-θ Monotonicity Law
- Geometric alignment follows: cos(θ_k) = c_0 + c_1(1 - e^{-k/τ})
- Validated across 15 models, K ∈ [1, 512], zero exceptions

### 3. Temperature Modulation: τ = τ_min + C/β
- Architecture-dependent temperature response
- MHA: Strong (C ≈ 27), GQA: Moderate (C ≈ 4), MQA: None (C ≈ 0)

### 4. SI Classification: Continuous Spectrum
- LP (Low Perturbation) → SP (Spectral Perturbation) → OSC (Oscillatory)
- No phase transitions, smooth continuous crossover

## Repository Structure

### Documentation
- [`docs/The_Transformer_Expansion_System.md`](docs/The_Transformer_Expansion_System.md) - Paper (English)
- [`docs/The_Transformer_Expansion_System_CN.md`](docs/The_Transformer_Expansion_System_CN.md) - Paper (Chinese)

### Scripts
- [`scripts/attention_temperature_enhanced_v1.py`](scripts/attention_temperature_enhanced_v1.py) - Temperature experiment
- [`scripts/full_block_jacobian_spectrum_test.py`](scripts/full_block_jacobian_spectrum_test.py) - Jacobian analysis
- [`scripts/calculate_si_all_models.py`](scripts/calculate_si_all_models.py) - SI calculation
- [`scripts/band_sensitivity_analysis.py`](scripts/band_sensitivity_analysis.py) - Sensitivity analysis
- [`scripts/pre_residual_control_experiment.py`](scripts/pre_residual_control_experiment.py) - Negative control
- [`scripts/negative_control_experiment.py`](scripts/negative_control_experiment.py) - Negative control
- [`scripts/tau_profile_likelihood.py`](scripts/tau_profile_likelihood.py) - Likelihood analysis
- [`scripts/analyze_k_star.py`](scripts/analyze_k_star.py) - K-θ analysis
- [`scripts/decisive_random_subspace_experiment.py`](scripts/decisive_random_subspace_experiment.py) - Decisive experiment
- [`scripts/common_utils.py`](scripts/common_utils.py) - Utility functions

### Validation Results
- [`validation_results/enhanced_temperature_20260310_195631.json`](validation_results/enhanced_temperature_20260310_195631.json) - Temperature data
- [`validation_results/summary.json`](validation_results/summary.json) - Alignment summary
- [`validation_results/pre_residual_control.json`](validation_results/pre_residual_control.json) - Control experiment
- [`validation_results/negative_control_perturbation.json`](validation_results/negative_control_perturbation.json) - Control experiment
- [`validation_results/k_star_validation_summary.json`](validation_results/k_star_validation_summary.json) - K-θ validation
- [`validation_results/profile_likelihood_summary.json`](validation_results/profile_likelihood_summary.json) - Likelihood summary
- [`validation_results/decisive_experiment_results.json`](validation_results/decisive_experiment_results.json) - Decisive experiment
- [`validation_results/figures/`](validation_results/figures/) - Visualization figures

## Models Studied

| Model Family | Architectures | Parameters |
|:-------------|:--------------|:-----------|
| Qwen2.5 | GQA | 0.5B, 1.5B, 3B, 7B |
| Phi-2 | MHA | 2.7B |
| Gemma | MQA | 2B |
| LLaMA | GQA | 7B |
| Pythia | MHA | Various |

## Quick Start

### Requirements

```bash
pip install torch transformers numpy scipy matplotlib
```

### Run Experiments

```python
# Example: Temperature modulation experiment
python scripts/attention_temperature_enhanced_v1.py

# Example: Jacobian spectrum analysis
python scripts/full_block_jacobian_spectrum_test.py
```

## Core Metrics

| Metric | Definition | Physical Meaning |
|:-------|:-----------|:-----------------|
| **λ** | Jacobian spectral radius | Expansion constant |
| **τ** | Mixing time | Alignment rate |
| **θ** | MLP-PCA angle | Geometric alignment |
| **SI** | Stability Index | Perturbation sensitivity |

## Citation

If you use this data or code, please cite:

```bibtex
@dataset{statelens_expansion_2026,
  title={The Transformer Expansion System: Geometry of Representation and Dynamics of Mixing},
  author={StateLens Project},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.xxxxx}
}
```

## License

MIT License

---

**Three-Axiom Framework**:

```
┌─────────────────────────────────────────────────────────────┐
│  AXIOM 1: Attention Determines Mixing Modes                 │
│  → Temperature β controls mixing time τ = τ_min + C/β      │
├─────────────────────────────────────────────────────────────┤
│  AXIOM 2: Embedding Determines Observable Modes             │
│  → Expansion constant λ ≈ 1.88 defines representation growth│
│  → K-θ monotonicity law governs geometric alignment         │
├─────────────────────────────────────────────────────────────┤
│  AXIOM 3: Logits Reflect Filtered Dynamics                  │
│  → Final output is low-dimensional projection after L layers│
└─────────────────────────────────────────────────────────────┘
```
