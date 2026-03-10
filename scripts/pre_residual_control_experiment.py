"""
Pre-Residual Control Experiment
================================

目标：排除残差连接的混淆变量

核心问题：
Reviewer 攻击："你的 alignment 测量的是 post-residual hidden state，
残差连接可能稀释了扰动效果，而不是真正的因果不变性。"

解决方案：
在 MLP 输出（pre-residual）上重新计算 alignment，
比较 perturbation 前后。

如果仍然 invariant → 立刻赢，Reviewer 无话可说。

测量位置对比：
- post-residual: x_out = x_in + MLP(x_in)  ← 原实验
- pre-residual:  h_mlp = MLP(x_in)         ← 本实验
"""

from __future__ import annotations

import math
import os
import random

os.environ["PYTORCH_JIT"] = "0"

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    model_path: Path
    test_layers: tuple[int, ...] = (2, 14)
    epsilon_values: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)
    n_random_samples: int = 100
    n_data_samples: int = 50
    n_permutations: int = 1000
    precision: str = "fp32"


def load_model(
    model_path: Path,
    precision: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype_map = {
        "fp64": torch.float64,
        "fp32": torch.float32,
    }
    dtype = dtype_map[precision]

    logger.info("Loading model from %s with precision %s", model_path, precision)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    return model, tokenizer, device, dtype


def compute_alignment(v1: Tensor, v2: Tensor) -> float:
    """
    Compute alignment (absolute cosine similarity).
    """
    v1 = v1.to(torch.float64).flatten()
    v2 = v2.to(torch.float64).flatten()
    v1_norm = v1 / v1.norm()
    v2_norm = v2 / v2.norm()
    return abs(torch.dot(v1_norm, v2_norm).item())


def compute_theoretical_baseline(dim: int) -> float:
    """
    Compute theoretical baseline for random alignment.
    E[|cos θ|] = sqrt(2 / (π * d))
    """
    return (2.0 / (math.pi * dim)) ** 0.5


def compute_perturbed_singular_vector(
    layer: torch.nn.Module,
    epsilon: float,
    perturb_target: str,
) -> tuple[Tensor, float]:
    """
    Compute singular vector of perturbed weight matrix.
    
    Perturbation: W_perturbed = W + ε * ||W|| * N(0, I)
    """
    w_gate = layer.mlp.gate_proj.weight.detach().float()
    w_down = layer.mlp.down_proj.weight.detach().float()
    
    if epsilon > 0:
        if perturb_target == "gate":
            w_norm = w_gate.norm()
            noise = torch.randn_like(w_gate)
            w_perturbed = w_gate + epsilon * w_norm * noise / noise.norm()
            w_eff = w_down @ w_perturbed
        elif perturb_target == "down":
            w_norm = w_down.norm()
            noise = torch.randn_like(w_down)
            w_perturbed = w_down + epsilon * w_norm * noise / noise.norm()
            w_eff = w_perturbed @ w_gate
        else:
            raise ValueError(f"Unknown perturb_target: {perturb_target}")
    else:
        w_eff = w_down @ w_gate
    
    U, S, Vh = torch.linalg.svd(w_eff, full_matrices=False)
    
    return Vh[0], S[0].item()


def capture_pre_residual_hidden_state(
    model: AutoModelForCausalLM,
    inputs: dict,
    layer_idx: int,
) -> Tensor:
    """
    Capture MLP output (pre-residual) hidden state.
    
    This measures h_mlp = MLP(LayerNorm(x_in)), NOT x_out = x_in + MLP(x_in).
    
    Key difference from post-residual:
    - post-residual: model.model.layers[layer_idx] output (includes residual)
    - pre-residual:  model.model.layers[layer_idx].mlp output (pure MLP)
    """
    hidden_states = []
    
    def hook_fn(module, args, output):
        if isinstance(output, tuple):
            h = output[0][0, -1]
        else:
            h = output[0, -1]
        hidden_states.append(h.detach())
    
    # 关键修改：hook 在 MLP 上，而不是整个层
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    
    return hidden_states[0]


def capture_post_residual_hidden_state(
    model: AutoModelForCausalLM,
    inputs: dict,
    layer_idx: int,
) -> Tensor:
    """
    Capture post-residual hidden state (for comparison).
    
    This measures x_out = x_in + MLP(x_in) + Attention(x_in).
    """
    hidden_states = []
    
    def hook_fn(module, args, output):
        if isinstance(output, tuple):
            h = output[0][0, -1]
        else:
            h = output[0, -1]
        hidden_states.append(h.detach())
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    
    return hidden_states[0]


def compute_matched_random_baseline(
    hidden_states: list[Tensor],
    singular_vector: Tensor,
    n_samples: int,
    device: torch.device,
) -> list[float]:
    """
    Compute matched random baseline alignment.
    """
    alignments = []
    dim = singular_vector.numel()
    
    for _ in range(n_samples):
        h = random.choice(hidden_states).to(device)
        h_norm = h.norm()
        
        r = torch.randn(dim, device=device)
        r = r / r.norm()
        
        h_random = h_norm * r
        alignment = compute_alignment(h_random, singular_vector)
        alignments.append(alignment)
    
    return alignments


def permutation_test(
    data_alignments: list[float],
    null_alignments: list[float],
    n_permutations: int = 1000,
) -> float:
    """
    Permutation test for significance.
    """
    observed_diff = abs(sum(data_alignments) / len(data_alignments) - sum(null_alignments) / len(null_alignments))
    
    combined = data_alignments + null_alignments
    n_data = len(data_alignments)
    
    count = 0
    for _ in range(n_permutations):
        random.shuffle(combined)
        perm_data = combined[:n_data]
        perm_null = combined[n_data:]
        perm_diff = abs(sum(perm_data) / n_data - sum(perm_null) / len(perm_null))
        if perm_diff >= observed_diff:
            count += 1
    
    return count / n_permutations


def compute_cohens_d(data_alignments: list[float], null_alignments: list[float]) -> float:
    """
    Compute Cohen's d effect size.
    """
    n1, n2 = len(data_alignments), len(null_alignments)
    mu1 = sum(data_alignments) / n1
    mu2 = sum(null_alignments) / n2
    
    var1 = sum((x - mu1) ** 2 for x in data_alignments) / n1
    var2 = sum((x - mu2) ** 2 for x in null_alignments) / n2
    
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return float('inf')
    
    return (mu1 - mu2) / pooled_std


def run_pre_residual_experiment(config: Config) -> dict:
    """
    Run pre-residual experiment to exclude residual connection confound.
    """
    model, tokenizer, device, dtype = load_model(config.model_path, config.precision)
    
    results = {}
    
    for layer_idx in config.test_layers:
        logger.info("=" * 60)
        logger.info("Layer %d", layer_idx)
        logger.info("=" * 60)
        
        layer = model.model.layers[layer_idx]
        
        # 1. Freeze hidden states (BOTH pre and post residual)
        logger.info("Sampling hidden states (frozen across all experiments)...")
        
        pre_residual_states = []
        post_residual_states = []
        
        for i in range(config.n_data_samples):
            prompt = ["Hello", "World", "Test", "AI", "ML"][i % 5]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            h_pre = capture_pre_residual_hidden_state(model, inputs, layer_idx)
            h_post = capture_post_residual_hidden_state(model, inputs, layer_idx)
            
            pre_residual_states.append(h_pre)
            post_residual_states.append(h_post)
        
        # 2. Compute baseline sigma_static at ε=0
        _, sigma_static_baseline = compute_perturbed_singular_vector(layer, 0.0, "gate")
        
        layer_results = {}
        
        # 3. Test both pre-residual and post-residual
        for state_type, hidden_states in [("pre_residual", pre_residual_states), 
                                           ("post_residual", post_residual_states)]:
            logger.info("")
            logger.info("Hidden state type: %s", state_type)
            logger.info("-" * 40)
            
            state_results = {}
            
            for perturb_target in ["gate", "down"]:
                logger.info("Perturbation target: %s", perturb_target.upper())
                
                target_results = {}
                
                for epsilon in config.epsilon_values:
                    singular_vector, sigma_static = compute_perturbed_singular_vector(
                        layer, epsilon, perturb_target
                    )
                    dim = singular_vector.numel()
                    theoretical_baseline = compute_theoretical_baseline(dim)
                    
                    data_alignments = [
                        compute_alignment(h, singular_vector)
                        for h in hidden_states
                    ]
                    
                    matched_alignments = compute_matched_random_baseline(
                        hidden_states, singular_vector, config.n_random_samples, device
                    )
                    
                    mean_data = sum(data_alignments) / len(data_alignments)
                    std_data = (sum((x - mean_data) ** 2 for x in data_alignments) / len(data_alignments)) ** 0.5
                    
                    mean_matched = sum(matched_alignments) / len(matched_alignments)
                    
                    p_value = permutation_test(data_alignments, matched_alignments, config.n_permutations)
                    cohens_d = compute_cohens_d(data_alignments, matched_alignments)
                    
                    relative_sigma = sigma_static / sigma_static_baseline if sigma_static_baseline > 0 else 1.0
                    
                    target_results[epsilon] = {
                        "sigma_static": sigma_static,
                        "relative_sigma": relative_sigma,
                        "alignment_mean": mean_data,
                        "alignment_std": std_data,
                        "matched_random_mean": mean_matched,
                        "theoretical_baseline": theoretical_baseline,
                        "alignment_ratio": mean_data / theoretical_baseline if theoretical_baseline > 0 else 0,
                        "p_value": p_value,
                        "cohens_d": cohens_d,
                    }
                    
                    logger.info("  ε=%.2f: alignment=%.4f±%.4f, Cohen's d=%.2f",
                               epsilon, mean_data, std_data, cohens_d)
                
                state_results[perturb_target] = target_results
            
            layer_results[state_type] = state_results
        
        results[layer_idx] = layer_results
    
    del model
    torch.cuda.empty_cache()
    
    return results


def print_summary(results: dict):
    """
    Print summary comparing pre-residual vs post-residual.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PRE-RESIDUAL vs POST-RESIDUAL COMPARISON")
    logger.info("=" * 80)
    
    for layer_idx, layer_results in results.items():
        logger.info("")
        logger.info("Layer %d:", layer_idx)
        logger.info("=" * 70)
        
        for state_type in ["pre_residual", "post_residual"]:
            if state_type not in layer_results:
                continue
            
            state_results = layer_results[state_type]
            logger.info("")
            logger.info("%s:", state_type.upper().replace("_", "-"))
            logger.info("-" * 70)
            
            for perturb_target in ["gate", "down"]:
                if perturb_target not in state_results:
                    continue
                
                target_results = state_results[perturb_target]
                alignments = [target_results[e]["alignment_mean"] for e in [0.0, 0.1]]
                cohens_ds = [target_results[e]["cohens_d"] for e in [0.0, 0.1]]
                
                decay = (alignments[0] - alignments[1]) / alignments[0] * 100
                
                logger.info("  %s: %.4f → %.4f (decay: %.1f%%), Cohen's d: %.2f → %.2f",
                           perturb_target.upper(),
                           alignments[0], alignments[1], decay,
                           cohens_ds[0], cohens_ds[1])
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("CAUSAL CONCLUSION")
    logger.info("=" * 80)
    
    for layer_idx, layer_results in results.items():
        logger.info("")
        logger.info("Layer %d:", layer_idx)
        
        pre = layer_results.get("pre_residual", {})
        post = layer_results.get("post_residual", {})
        
        if "gate" in pre and "gate" in post:
            pre_decay = (pre["gate"][0.0]["alignment_mean"] - 
                        pre["gate"][0.1]["alignment_mean"]) / pre["gate"][0.0]["alignment_mean"] * 100
            post_decay = (post["gate"][0.0]["alignment_mean"] - 
                         post["gate"][0.1]["alignment_mean"]) / post["gate"][0.0]["alignment_mean"] * 100
            
            logger.info("  Pre-residual decay:  %.1f%%", pre_decay)
            logger.info("  Post-residual decay: %.1f%%", post_decay)
            
            if abs(pre_decay) < 10 and abs(post_decay) < 10:
                logger.info("  → BOTH invariant: Residual connection is NOT a confound.")
                logger.info("     Alignment is genuinely robust to MLP perturbations.")
            elif abs(pre_decay) > 20 and abs(post_decay) < 10:
                logger.info("  → Pre-residual sensitive, post-residual invariant.")
                logger.info("     Residual connection DOES dilute the perturbation effect.")
                logger.info("     Causal conclusion needs revision.")
            else:
                logger.info("  → Mixed pattern: Further analysis needed.")


def main():
    import json
    from datetime import datetime
    
    config = Config(
        model_path=Path(r"E:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\Qwen2.5-0.5B"),
        test_layers=(2, 14),
        epsilon_values=(0.0, 0.01, 0.05, 0.1),
        n_random_samples=100,
        n_data_samples=50,
        n_permutations=1000,
        precision="fp32",
    )
    
    results = run_pre_residual_experiment(config)
    print_summary(results)
    
    output_data = {
        "experiment": "pre_residual_control",
        "model": config.model_path.name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "排除残差连接的混淆变量",
        "perturbation_method": "W_perturbed = W + ε * ||W|| * N(0, I)",
        "epsilon_values": list(config.epsilon_values),
        "config": {
            "test_layers": config.test_layers,
            "n_random_samples": config.n_random_samples,
            "n_data_samples": config.n_data_samples,
            "n_permutations": config.n_permutations,
            "precision": config.precision,
        },
        "layers": {},
        "key_findings": [],
        "final_conclusion": ""
    }
    
    for layer_idx, layer_results in results.items():
        layer_key = f"layer_{layer_idx}"
        output_data["layers"][layer_key] = {}
        
        for state_type in ["pre_residual", "post_residual"]:
            if state_type not in layer_results:
                continue
            
            state_results = layer_results[state_type]
            output_data["layers"][layer_key][state_type] = {}
            
            for perturb_target in ["gate", "down"]:
                if perturb_target not in state_results:
                    continue
                
                target_results = state_results[perturb_target]
                epsilons = list(target_results.keys())
                alignments = [target_results[e]["alignment_mean"] for e in epsilons]
                cohens_ds = [target_results[e]["cohens_d"] for e in epsilons]
                
                alignment_decay = (alignments[0] - alignments[-1]) / alignments[0] * 100 if alignments[0] != 0 else 0
                
                output_data["layers"][layer_key][state_type][perturb_target.upper()] = {
                    "epsilon_0.0": target_results[0.0]["alignment_mean"],
                    "epsilon_0.1": target_results[0.1]["alignment_mean"],
                    "decay_percent": round(alignment_decay, 1),
                    "cohens_d_before": round(cohens_ds[0], 2),
                    "cohens_d_after": round(cohens_ds[-1], 2),
                    "full_results": {str(k): v for k, v in target_results.items()}
                }
    
    output_path = Path("validation_results/pre_residual_control.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
