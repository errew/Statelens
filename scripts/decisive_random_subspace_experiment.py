# -*- coding: utf-8 -*-
"""
Decisive Experiment: Random Subspace Baseline
==============================================

解决质疑者的三个关键问题：
Q1: θ_random baseline? - 如果random subspace本来就是80°，结果没意义
Q2: θ across layers? - 是否layer dependent
Q3: dataset PCA vs prompt PCA? - 是否一致

决定性实验：
- 正确PCA：1000句子 → 100k tokens → H shape = [100k, hidden_dim]
- Random subspace baseline: V_random ∈ R^(d×K)
- 对比 θ_model vs θ_random

如果 θ_model << θ_random → 真实耦合
如果 θ_model ≈ θ_random → 纯高维几何幻觉
"""
import torch
import gc
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("DecisiveExp")

MODEL_PATH = Path(r"E:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\Qwen2.5-0.5B")
RESULTS_DIR = Path("f:/Codes/comfyui-statelens/validation_results/decisive_experiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SENTENCES = 1000
K_VALUES = [1, 5, 10, 20, 50, 100]
N_RANDOM_BASELINES = 100


@dataclass
class ExperimentConfig:
    n_sentences: int = 1000
    n_random_baselines: int = 100
    k_values: Tuple[int, ...] = (1, 5, 10, 20, 50, 100)


def generate_diverse_sentences(n: int) -> List[str]:
    templates = [
        "The {} of {} is {}.",
        "In the {} {}, we found {}.",
        "Research shows that {} {} {}.",
        "The study of {} reveals {}.",
        "Scientists discovered {} in {}.",
        "The analysis of {} indicates {}.",
        "According to {} theory, {}.",
        "The relationship between {} and {} is {}.",
        "Experimental results show {}.",
        "The hypothesis that {} was {}.",
        "Data suggests {} correlation with {}.",
        "The mechanism of {} involves {}.",
        "Historical records indicate {}.",
        "Statistical analysis reveals {}.",
        "The phenomenon of {} occurs when {}.",
        "Mathematical models predict {}.",
        "The system exhibits {} behavior.",
        "Observations confirm {}.",
        "The process of {} requires {}.",
        "Complex interactions lead to {}.",
    ]
    
    subjects = [
        "neural networks", "quantum mechanics", "machine learning", "deep learning",
        "natural language", "computer vision", "data science", "artificial intelligence",
        "statistical models", "optimization algorithms", "transformer architecture",
        "attention mechanism", "gradient descent", "backpropagation", "feature extraction",
        "dimensionality reduction", "clustering algorithms", "classification tasks",
        "regression analysis", "time series", "signal processing", "information theory",
        "cryptography", "distributed systems", "parallel computing", "memory management",
        "database systems", "network protocols", "operating systems", "compiler design",
    ]
    
    adjectives = [
        "significant", "remarkable", "interesting", "complex", "fundamental",
        "essential", "critical", "notable", "surprising", "unexpected",
        "consistent", "variable", "stable", "dynamic", "emergent",
    ]
    
    outcomes = [
        "improved performance", "new insights", "unexpected results",
        "significant correlations", "novel patterns", "complex interactions",
        "stable equilibria", "dynamic transitions", "emergent properties",
        "fundamental principles", "practical applications", "theoretical implications",
    ]
    
    sentences = []
    for _ in range(n):
        template = random.choice(templates)
        sentence = template.format(
            random.choice(subjects),
            random.choice(adjectives),
            random.choice(outcomes),
        )
        sentences.append(sentence)
    
    return sentences


def get_mlp_weight_combined(layer) -> Optional[torch.Tensor]:
    mlp = layer.mlp if hasattr(layer, 'mlp') else layer
    
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
        Wg = mlp.gate_proj.weight.detach().float().cpu()
        Wd = mlp.down_proj.weight.detach().float().cpu()
        if hasattr(mlp, "up_proj"):
            Wu = mlp.up_proj.weight.detach().float().cpu()
            return Wd @ (Wg * Wu)
        return Wd @ Wg
    elif hasattr(mlp, "down_proj"):
        return mlp.down_proj.weight.detach().float().cpu()
    return None


def get_attn_weight(layer) -> Optional[torch.Tensor]:
    attn = layer.self_attn if hasattr(layer, 'self_attn') else None
    if attn is None:
        return None
    if hasattr(attn, "o_proj"):
        return attn.o_proj.weight.detach().float().cpu()
    return None


def compute_principal_angle(weight: torch.Tensor, V_k: torch.Tensor) -> float:
    _, _, Vh_weight = torch.linalg.svd(weight.to(torch.float64), full_matrices=False)
    direction = Vh_weight[0].flatten()
    direction = direction / (direction.norm() + 1e-8)
    
    V_k = V_k.to(torch.float64)
    
    hidden_dim = V_k.shape[0]
    weight_dim = direction.shape[0]
    
    if hidden_dim != weight_dim:
        min_dim = min(hidden_dim, weight_dim)
        if hidden_dim > weight_dim:
            V_k = V_k[:min_dim, :]
        else:
            direction = direction[:min_dim]
    
    proj_coeffs = V_k.T @ direction
    proj_norm_sq = (proj_coeffs.norm() ** 2).item()
    
    cos_val = math.sqrt(max(0.0, min(1.0, proj_norm_sq)))
    return math.degrees(math.acos(cos_val))


def generate_random_subspace(hidden_dim: int, k: int) -> torch.Tensor:
    V_random = torch.randn(hidden_dim, k)
    Q, _ = torch.linalg.qr(V_random)
    return Q


def compute_random_baseline_angle(weight: torch.Tensor, hidden_dim: int, k: int, 
                                   n_samples: int = 100) -> Tuple[float, float]:
    angles = []
    for _ in range(n_samples):
        V_random = generate_random_subspace(hidden_dim, k)
        angle = compute_principal_angle(weight, V_random)
        angles.append(angle)
    
    mean_angle = sum(angles) / len(angles)
    std_angle = math.sqrt(sum((a - mean_angle)**2 for a in angles) / len(angles))
    return mean_angle, std_angle


def collect_hidden_states_correct(model, tokenizer, sentences: List[str], 
                                   device: torch.device) -> Dict[int, torch.Tensor]:
    model.eval()
    n_layers = len(model.model.layers)
    
    layer_hidden_states: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}
    
    logger.info(f"Collecting hidden states from {len(sentences)} sentences...")
    
    with torch.no_grad():
        for i, sentence in enumerate(sentences):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processing sentence {i+1}/{len(sentences)}")
            
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            for layer_idx in range(n_layers):
                hidden = outputs.hidden_states[layer_idx][0].cpu().float()
                layer_hidden_states[layer_idx].append(hidden)
    
    logger.info("Concatenating hidden states...")
    result = {}
    for layer_idx in range(n_layers):
        H = torch.cat(layer_hidden_states[layer_idx], dim=0)
        result[layer_idx] = H
        logger.info(f"  Layer {layer_idx}: H shape = {H.shape}")
    
    return result


def analyze_layer(H: torch.Tensor, W_mlp: torch.Tensor, W_attn: torch.Tensor,
                  k_values: List[int], n_random: int = 100) -> Dict:
    H_centered = H - H.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    
    actual_rank = min(H.shape[0], H.shape[1])
    logger.info(f"  Actual rank: min({H.shape[0]}, {H.shape[1]}) = {actual_rank}")
    
    hidden_dim = Vh.shape[1]
    
    results = {
        "n_tokens": H.shape[0],
        "hidden_dim": hidden_dim,
        "actual_rank": actual_rank,
        "angles_model": {},
        "angles_random": {},
        "coupling_ratio": {},
    }
    
    for k in k_values:
        if k > actual_rank:
            logger.info(f"  K={k} > rank={actual_rank}, skipping")
            continue
        
        V_k = Vh[:k, :].T
        
        theta_mlp = compute_principal_angle(W_mlp, V_k)
        theta_attn = compute_principal_angle(W_attn, V_k)
        theta_model = min(theta_mlp, theta_attn)
        
        theta_random_mean, theta_random_std = compute_random_baseline_angle(
            W_mlp, hidden_dim, k, n_random
        )
        
        coupling_ratio = theta_model / theta_random_mean if theta_random_mean > 0 else float('nan')
        
        results["angles_model"][k] = {
            "theta_mlp": theta_mlp,
            "theta_attn": theta_attn,
            "theta_min": theta_model,
        }
        results["angles_random"][k] = {
            "mean": theta_random_mean,
            "std": theta_random_std,
        }
        results["coupling_ratio"][k] = coupling_ratio
        
        logger.info(f"  K={k}: θ_model={theta_model:.1f}°, θ_random={theta_random_mean:.1f}°±{theta_random_std:.1f}°, "
                    f"ratio={coupling_ratio:.3f}")
    
    return results


def run_decisive_experiment(config: ExperimentConfig) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()
    
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    logger.info(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")
    
    sentences = generate_diverse_sentences(config.n_sentences)
    logger.info(f"Generated {len(sentences)} sentences")
    
    layer_hidden_states = collect_hidden_states_correct(model, tokenizer, sentences, device)
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    test_layers = [l for l in test_layers if l < n_layers]
    
    layer_results = {}
    
    for layer_idx in test_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing Layer {layer_idx}")
        logger.info(f"{'='*60}")
        
        H = layer_hidden_states[layer_idx]
        
        W_mlp = get_mlp_weight_combined(model.model.layers[layer_idx])
        W_attn = get_attn_weight(model.model.layers[layer_idx])
        
        if W_mlp is None or W_attn is None:
            logger.warning(f"  Skipping layer {layer_idx}: missing weights")
            continue
        
        result = analyze_layer(H, W_mlp, W_attn, list(config.k_values), config.n_random_baselines)
        layer_results[layer_idx] = result
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "config": {
            "n_sentences": config.n_sentences,
            "n_random_baselines": config.n_random_baselines,
            "k_values": list(config.k_values),
        },
        "model_info": {
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
        },
        "layer_results": layer_results,
    }


def analyze_results(results: Dict) -> Dict:
    logger.info("\n" + "="*70)
    logger.info("DECISIVE EXPERIMENT ANALYSIS")
    logger.info("="*70)
    
    analysis = {
        "summary": {},
        "conclusions": {},
    }
    
    for layer_idx, layer_data in results["layer_results"].items():
        logger.info(f"\nLayer {layer_idx}:")
        logger.info(f"  N tokens: {layer_data['n_tokens']}")
        logger.info(f"  Hidden dim: {layer_data['hidden_dim']}")
        logger.info(f"  Actual rank: {layer_data['actual_rank']}")
        
        for k, ratio in layer_data["coupling_ratio"].items():
            if math.isnan(ratio):
                continue
            
            theta_model = layer_data["angles_model"][k]["theta_min"]
            theta_random = layer_data["angles_random"][k]["mean"]
            
            if ratio < 0.9:
                verdict = "TRUE COUPLING"
            elif ratio > 1.1:
                verdict = "ANTI-COUPLING"
            else:
                verdict = "RANDOM-LIKE"
            
            logger.info(f"  K={k}: θ_model={theta_model:.1f}°, θ_random={theta_random:.1f}°, "
                        f"ratio={ratio:.3f} → {verdict}")
    
    all_ratios_k100 = []
    for layer_data in results["layer_results"].values():
        if 100 in layer_data["coupling_ratio"]:
            ratio = layer_data["coupling_ratio"][100]
            if not math.isnan(ratio):
                all_ratios_k100.append(ratio)
    
    if all_ratios_k100:
        mean_ratio = sum(all_ratios_k100) / len(all_ratios_k100)
        min_ratio = min(all_ratios_k100)
        max_ratio = max(all_ratios_k100)
        
        n_coupling = sum(1 for r in all_ratios_k100 if r < 0.9)
        n_random = sum(1 for r in all_ratios_k100 if 0.9 <= r <= 1.1)
        n_anti = sum(1 for r in all_ratios_k100 if r > 1.1)
        
        coupling_ratios = [r for r in all_ratios_k100 if r < 0.9]
        mean_coupling_only = sum(coupling_ratios) / len(coupling_ratios) if coupling_ratios else float('nan')
        
        logger.info(f"\n{'='*70}")
        logger.info("STATISTICAL SUMMARY (K=100)")
        logger.info(f"{'='*70}")
        logger.info(f"  Mean ratio (all layers): {mean_ratio:.3f}")
        logger.info(f"  Min ratio (strongest coupling): {min_ratio:.3f}")
        logger.info(f"  Max ratio: {max_ratio:.3f}")
        logger.info(f"  Mean ratio (coupling layers only): {mean_coupling_only:.3f}")
        logger.info(f"\n  Layer distribution:")
        logger.info(f"    TRUE COUPLING (ratio < 0.9): {n_coupling}/{len(all_ratios_k100)}")
        logger.info(f"    RANDOM-LIKE (0.9 ≤ ratio ≤ 1.1): {n_random}/{len(all_ratios_k100)}")
        logger.info(f"    ANTI-COUPLING (ratio > 1.1): {n_anti}/{len(all_ratios_k100)}")
        
        logger.info(f"\n  Layer-wise histogram:")
        for i, r in enumerate(sorted(all_ratios_k100)):
            bar = "█" * int(r * 10)
            logger.info(f"    [{i+1}] ratio={r:.3f} {bar}")
        
        if min_ratio < 0.7:
            conclusion = "STRONG COUPLING: min_ratio < 0.7, θ_model << θ_random in middle layers"
            logger.info(f"\nCONCLUSION: {conclusion}")
        elif min_ratio < 0.9:
            conclusion = "MODERATE COUPLING: min_ratio < 0.9"
            logger.info(f"\nCONCLUSION: {conclusion}")
        elif mean_ratio > 1.1:
            conclusion = "ANTI-COUPLING: θ_model >> θ_random"
            logger.info(f"\nCONCLUSION: {conclusion}")
        else:
            conclusion = "RANDOM-LIKE: θ_model ≈ θ_random (高维几何幻觉)"
            logger.info(f"\nCONCLUSION: {conclusion}")
        
        analysis["conclusions"]["k100"] = {
            "mean_ratio": mean_ratio,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "mean_coupling_only": mean_coupling_only,
            "n_coupling_layers": n_coupling,
            "n_random_layers": n_random,
            "n_anti_layers": n_anti,
            "total_layers": len(all_ratios_k100),
            "verdict": conclusion,
        }
    
    return analysis


def main():
    logger.info("="*70)
    logger.info("DECISIVE EXPERIMENT: Random Subspace Baseline")
    logger.info("="*70)
    logger.info("解决三个关键问题:")
    logger.info("Q1: θ_random baseline?")
    logger.info("Q2: θ across layers?")
    logger.info("Q3: dataset PCA vs prompt PCA?")
    logger.info("="*70)
    
    config = ExperimentConfig(
        n_sentences=N_SENTENCES,
        n_random_baselines=N_RANDOM_BASELINES,
        k_values=tuple(K_VALUES),
    )
    
    results = run_decisive_experiment(config)
    analysis = analyze_results(results)
    
    results["analysis"] = analysis
    
    output_file = RESULTS_DIR / "decisive_experiment_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
