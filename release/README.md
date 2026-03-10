# StateLens: The Transformer Expansion System

> **Attention determines mixing modes, embedding determines observable modes, logits reflect filtered dynamics.**

## Overview

This repository contains the experimental data and analysis code for our comprehensive study of Transformer internal dynamics. We present evidence that Transformer layers operate as **expansive systems** (Layer 2 λ ≈ 1.7-1.8 > 1), challenging the conventional assumption of contractive representation learning.

## Key Findings

### 1. Transformer Layers are Expansive Systems
- Spectral radius Layer 2 λ ≈ 1.7-1.8 > 1 across 15+ models
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

```
statelens/
├── docs/
│   ├── The_Transformer_Expansion_System.md      # Paper (English)
│   └── The_Transformer_Expansion_System_CN.md   # Paper (Chinese)
├── scripts/
│   ├── attention_temperature_enhanced_v1.py     # Temperature experiment
│   ├── full_block_jacobian_spectrum_test.py     # Jacobian analysis
│   ├── calculate_si_all_models.py               # SI calculation
│   ├── band_sensitivity_analysis.py             # Sensitivity analysis
│   ├── pre_residual_control_experiment.py       # Negative control
│   ├── negative_control_experiment.py           # Negative control
│   ├── decisive_random_subspace_experiment.py   # Causal validation
│   └── tau_profile_likelihood.py                # Likelihood analysis
├── validation_results/
│   ├── layer2_jacobian_spectral_analysis.json   # Layer 2 spectral data
│   ├── layer2_spectral_results_20260311.json    # Multi-model results
│   ├── enhanced_temperature_20260310_195631.json # Temperature data
│   ├── k_star_validation_summary.json           # K-θ validation
│   ├── decisive_experiment_results.json         # Decisive experiment
│   ├── pre_residual_control.json                # Control experiment
│   ├── negative_control_perturbation.json       # Control experiment
│   └── profile_likelihood_summary.json          # Likelihood summary
└── README.md
```

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

## Contact

- Issues: [GitHub Issues](https://github.com/xxx/statelens/issues)
- Discussions: [GitHub Discussions](https://github.com/xxx/statelens/discussions)

---

**Three-Axiom Framework**:

```
┌─────────────────────────────────────────────────────────────┐
│  AXIOM 1: Attention Determines Mixing Modes                 │
│  → Temperature β controls mixing time τ = τ_min + C/β      │
├─────────────────────────────────────────────────────────────┤
│  AXIOM 2: Embedding Determines Observable Modes             │
│  → Expansion constant λ ≈ 1.7-1.8 (Layer 2) defines growth  │
│  → K-θ monotonicity law governs geometric alignment         │
├─────────────────────────────────────────────────────────────┤
│  AXIOM 3: Logits Reflect Filtered Dynamics                  │
│  → Final output is low-dimensional projection after L layers│
└─────────────────────────────────────────────────────────────┘
```
