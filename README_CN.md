# StateLens: Transformer 扩张系统

> **注意力决定混合模式，嵌入决定可观测模式，Logits 反映过滤后的动力学。**

## 概述

本仓库包含我们对 Transformer 内部动力学进行综合研究的实验数据和分析代码。我们提供的证据表明，Transformer 层作为**扩张系统**运行（λ ≈ 1.88 > 1），挑战了收缩表征学习的传统假设。

## 核心发现

### 1. Transformer 层是扩张系统
- 15+ 模型的谱半径 λ ≈ 1.88 > 1
- 稳定性来自 LayerNorm，而非权重收缩

### 2. K-θ 单调性定律
- 几何对齐遵循：cos(θ_k) = c_0 + c_1(1 - e^{-k/τ})
- 在 15 个模型上验证，K ∈ [1, 512]，零例外

### 3. 温度调制：τ = τ_min + C/β
- 架构依赖的温度响应
- MHA：强（C ≈ 27），GQA：中等（C ≈ 4），MQA：无（C ≈ 0）

### 4. SI 分类：连续谱
- LP（低扰动区）→ SP（谱扰动区）→ OSC（振荡区）
- 无相变，平滑连续过渡

## 仓库结构

```
statelens/
├── docs/
│   ├── The_Transformer_Expansion_System.md      # 论文（英文）
│   ├── The_Transformer_Expansion_System_CN.md   # 论文（中文）
│   ├── experiments/
│   │   └── 01_VALIDATION_SUMMARY.md             # 实验汇总
│   └── theory/
│       └── 01_CORE_THEORY.md                    # 核心理论文档
├── scripts/
│   ├── attention_temperature_enhanced_v1.py     # 温度实验
│   ├── full_block_jacobian_spectrum_test.py     # Jacobian 分析
│   ├── calculate_si_all_models.py               # SI 计算
│   └── ...
├── validation_results/
│   ├── attention_temperature_enhanced/          # 温度数据
│   ├── optimized_alignment/                     # 对齐数据
│   └── ...
└── README.md
```

## 研究模型

| 模型系列 | 架构 | 参数量 |
|:---------|:-----|:-------|
| Qwen2.5 | GQA | 0.5B, 1.5B, 3B, 7B |
| Phi-2 | MHA | 2.7B |
| Gemma | MQA | 2B |
| LLaMA | GQA | 7B |
| Pythia | MHA | 多种 |

## 快速开始

### 环境要求

```bash
pip install torch transformers numpy scipy matplotlib
```

### 运行实验

```python
# 示例：温度调制实验
python scripts/attention_temperature_enhanced_v1.py

# 示例：Jacobian 谱分析
python scripts/full_block_jacobian_spectrum_test.py
```

## 核心指标

| 指标 | 定义 | 物理意义 |
|:-----|:-----|:---------|
| **λ** | Jacobian 谱半径 | 扩张常数 |
| **τ** | 混合时间 | 对齐速率 |
| **θ** | MLP-PCA 角度 | 几何对齐 |
| **SI** | 稳定性指数 | 扰动敏感性 |

## 引用

如果您使用本数据或代码，请引用：

```bibtex
@dataset{statelens_expansion_2026,
  title={The Transformer Expansion System: Geometry of Representation and Dynamics of Mixing},
  author={StateLens Project},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.xxxxx}
}
```

## 许可证

MIT License

---

**三公理框架**：

```
┌─────────────────────────────────────────────────────────────┐
│  公理 1：注意力决定混合模式                                  │
│  → 温度 β 控制混合时间 τ = τ_min + C/β                      │
├─────────────────────────────────────────────────────────────┤
│  公理 2：嵌入决定可观测模式                                  │
│  → 扩张常数 λ ≈ 1.88 定义表征增长                           │
│  → K-θ 单调性定律支配几何对齐                               │
├─────────────────────────────────────────────────────────────┤
│  公理 3：Logits 反映过滤后的动力学                          │
│  → 最终输出是经过 L 层后的低维投影                          │
└─────────────────────────────────────────────────────────────┘
```
