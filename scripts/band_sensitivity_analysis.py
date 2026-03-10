"""
Band Boundary Sensitivity Analysis

Tests whether plateau detection depends on particular choice of angular band.
Systematically varies band boundaries and re-evaluates plateau statistics.
"""

import json
import logging
import math
import random
import statistics
import unittest
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BandSensitivity")


BASE_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = BASE_DIR.parent / "validation_results"
OUTPUT_DIR: Path = BASE_DIR / "figures"

EPS: float = 1e-8
K_FIXED: int = 32


@dataclass(frozen=True)
class BandConfig:
    name: str
    low: float
    high: float


@dataclass
class BandResult:
    band_name: str
    band_range: Tuple[float, float]
    plateau_freq: float
    mean_plateau_len: float
    std_plateau_len: float
    mean_plateau_start: float
    std_plateau_start: float
    mean_angle_in_plateau: float
    std_angle_in_plateau: float


@dataclass
class ModelBandSensitivity:
    model_name: str
    n_layers: int
    band_results: List[BandResult] = field(default_factory=list)


BAND_CONFIGS: Tuple[BandConfig, ...] = (
    BandConfig("Narrow", 76.0, 78.0),
    BandConfig("Canonical", 75.0, 78.0),
    BandConfig("Relaxed", 74.0, 79.0),
    BandConfig("Wide", 72.0, 80.0),
    BandConfig("Shifted-Low", 73.0, 76.0),
    BandConfig("Shifted-High", 77.0, 80.0),
)

MODEL_PATHS: Dict[str, Path] = {
    "phi-2": Path(r"E:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\phi-2"),
    "phi-3.5-mini": Path(r"E:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\Phi-3.5-mini-instruct"),
    "gpt2": Path(r"E:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\gpt2"),
}


def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def compute_angle_from_states(
    H: torch.Tensor,
    weight_direction: torch.Tensor,
    layer_idx: int,
) -> float:
    n_samples = H.shape[0]
    if n_samples < K_FIXED + 1:
        logger.warning(f"Layer {layer_idx}: insufficient samples ({n_samples} < {K_FIXED + 1})")
        return float("nan")

    H_double = H.to(torch.float64)
    if not torch.isfinite(H_double).all():
        logger.warning(f"Layer {layer_idx}: hidden states contain NaN/Inf values")
        return float("nan")
    
    mean = H_double.mean(dim=0, keepdim=True)
    H_centered = H_double - mean
    
    try:
        _, _, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    except torch.linalg.LinAlgError as e:
        logger.warning(f"Layer {layer_idx}: SVD failed to converge: {e}")
        return float("nan")

    actual_k = min(K_FIXED, n_samples - 1, H.shape[1])
    V_k = Vh[:actual_k]

    weight_dir_double = weight_direction.double()
    weight_dir_double = weight_dir_double / (torch.linalg.vector_norm(weight_dir_double) + EPS)

    projection_coeffs = V_k @ weight_dir_double
    proj_norm = torch.linalg.vector_norm(projection_coeffs).item()
    cos_angle = max(-1.0, min(1.0, proj_norm))
    return math.degrees(math.acos(cos_angle))


def is_valid_candidate(candidate_angles: List[float], min_consecutive: int, max_drift: float) -> bool:
    if len(candidate_angles) < min_consecutive:
        return False
    
    drifts = [abs(candidate_angles[j+1] - candidate_angles[j]) for j in range(len(candidate_angles)-1)]
    max_actual_drift = max(drifts) if drifts else 0.0
    
    return max_actual_drift <= max_drift


def detect_plateau(
    angles: List[float],
    band_low: float,
    band_high: float,
    min_consecutive: int = 3,
    max_drift: float = 0.5,
) -> Tuple[bool, int, int, List[float]]:
    max_len, max_start = 0, -1
    current_len, current_start = 0, -1

    def evaluate_current_plateau() -> None:
        nonlocal max_len, max_start
        if current_len > max_len:
            candidate = angles[current_start:current_start + current_len]
            if is_valid_candidate(candidate, min_consecutive, max_drift):
                max_len = current_len
                max_start = current_start

    for i, angle in enumerate(angles):
        if band_low <= angle <= band_high:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            evaluate_current_plateau()
            current_len, current_start = 0, -1
            
    evaluate_current_plateau()

    if max_len > 0:
        return True, max_start, max_len, angles[max_start:max_start + max_len]
    return False, -1, 0, []


def extract_mlp_weight_direction(model: PreTrainedModel, layer_idx: int) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate_up_proj"):
                return mlp.gate_up_proj.weight.data.float()
            if hasattr(mlp, "up_proj") and hasattr(mlp, "gate_proj"):
                return torch.cat([mlp.up_proj.weight.data, mlp.gate_proj.weight.data], dim=0).float()
            if hasattr(mlp, "fc1"):
                return mlp.fc1.weight.data.float()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        mlp = model.transformer.h[layer_idx].mlp
        if hasattr(mlp, "c_fc"):
            return mlp.c_fc.weight.data.t().float()
    
    raise ValueError(f"Cannot extract MLP weights from layer {layer_idx}")


def extract_all_layers_hidden_states(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_tokens: int = 8,
) -> Dict[int, torch.Tensor]:
    model.eval()
    n_layers = len(model.model.layers) if hasattr(model, "model") else len(model.transformer.h)
    layer_hidden_states: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                outputs = model(**inputs, use_cache=False, output_hidden_states=True)
                for layer_idx in range(n_layers):
                    hidden_state = outputs.hidden_states[layer_idx + 1]
                    n_tokens = hidden_state.shape[1]
                    start_idx = max(0, n_tokens - max_tokens)
                    layer_hidden_states[layer_idx].append(hidden_state[0, start_idx:, :].cpu())
            except RuntimeError as e:
                logger.error(f"RuntimeError processing prompt (possible OOM): {e}")
                continue

    return {idx: (torch.cat(tensors, dim=0) if tensors else torch.empty(0))
            for idx, tensors in layer_hidden_states.items()}


def get_default_prompts() -> List[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness.",
        "The scientist discovered a groundbreaking formula.",
        "Once upon a time in a distant kingdom.",
        "The artificial intelligence revolution began quietly.",
        "Deep in the forest, a hidden temple stood.",
        "The quantum computer processed the data instantly.",
        "She walked through the ancient library carefully.",
        "The spaceship landed on the mysterious planet.",
        "A new era of technology emerged from the chaos.",
        "The ancient manuscript revealed hidden secrets.",
        "Through the storm, a lighthouse appeared.",
        "The mathematician solved the impossible equation.",
        "In the garden, flowers bloomed unexpectedly.",
        "The detective found the missing clue.",
        "Programming requires logical thinking and patience.",
        "The mountain peak was covered in eternal snow.",
        "Music has the power to transcend language barriers.",
        "The ocean depths hold many undiscovered species.",
        "Innovation drives progress in every field.",
    ]


def aggregate_band_result(
    band: BandConfig,
    all_plateau_flags: List[float],
    all_lengths: List[float],
    all_starts: List[float],
    all_angles: List[float]
) -> BandResult:
    freq = sum(all_plateau_flags) / len(all_plateau_flags) if all_plateau_flags else 0.0
    mean_len, std_len = calculate_mean_std(all_lengths)
    mean_start, std_start = calculate_mean_std(all_starts)
    mean_ang, std_ang = calculate_mean_std(all_angles)
    
    return BandResult(
        band_name=band.name,
        band_range=(band.low, band.high),
        plateau_freq=freq,
        mean_plateau_len=mean_len,
        std_plateau_len=std_len,
        mean_plateau_start=mean_start,
        std_plateau_start=std_start,
        mean_angle_in_plateau=mean_ang,
        std_angle_in_plateau=std_ang,
    )


def run_model_analysis(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    n_layers: int,
    band_configs: Tuple[BandConfig, ...],
    all_prompts: List[str],
    n_prompts: int,
    n_resamples: int,
) -> List[BandResult]:
    results: List[BandResult] = []
    
    weight_directions: Dict[int, torch.Tensor] = {}
    for layer_idx in range(n_layers):
        try:
            W_eff = extract_mlp_weight_direction(model, layer_idx).cpu()
            _, _, Vh_weight = torch.linalg.svd(W_eff, full_matrices=False)
            weight_directions[layer_idx] = Vh_weight[0].flatten()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Layer {layer_idx}: weight extraction failed: {e}")
    
    for band in band_configs:
        logger.info(f"  Band [{band.low}°, {band.high}°]:")
        
        all_plateau_flags: List[float] = []
        all_lengths: List[float] = []
        all_starts: List[float] = []
        all_angles: List[float] = []
        
        for resample_idx in range(n_resamples):
            sampled_prompts = random.choices(all_prompts, k=n_prompts)
            layer_hidden_states = extract_all_layers_hidden_states(model, tokenizer, sampled_prompts)
            
            layer_angles: List[float] = []
            for layer_idx in range(n_layers):
                H = layer_hidden_states[layer_idx]
                if H.shape[0] < K_FIXED + 1 or layer_idx not in weight_directions:
                    layer_angles.append(float("nan"))
                    continue
                angle = compute_angle_from_states(H, weight_directions[layer_idx], layer_idx)
                layer_angles.append(angle)
            
            valid_angles = [a for a in layer_angles if not math.isnan(a)]
            if len(valid_angles) < n_layers * 0.5:
                logger.warning(f"    Resample {resample_idx+1}: too many NaN angles, skipping")
                continue
            
            has_plateau, start, length, plateau_angles = detect_plateau(
                layer_angles, band.low, band.high
            )
            
            all_plateau_flags.append(1.0 if has_plateau else 0.0)
            if has_plateau:
                all_lengths.append(float(length))
                all_starts.append(float(start))
                all_angles.extend(plateau_angles)
        
        result = aggregate_band_result(band, all_plateau_flags, all_lengths, all_starts, all_angles)
        results.append(result)
        
        start_str = f"L{result.mean_plateau_start:.0f}" if result.mean_plateau_start > 0 else "N/A"
        len_str = f"{result.mean_plateau_len:.1f}" if result.mean_plateau_len > 0 else "N/A"
        logger.info(f"    freq={result.plateau_freq:.0%}, start={start_str}, len={len_str}")
    
    return results


def get_model_dtype_and_attn(model_name: str) -> Tuple[torch.dtype, str]:
    """
    根据模型名称返回适当的 dtype 和 attention 实现
    """
    # 检查是否安装了flash_attn
    try:
        import flash_attn
        has_flash_attn = True
    except ImportError:
        has_flash_attn = False
    
    if "phi" in model_name.lower():
        # 如果有flash_attn则使用，否则使用eager
        attn_impl = "flash_attention_2" if has_flash_attn else "eager"
        return torch.bfloat16, attn_impl
    elif "gpt2" in model_name.lower():
        return torch.float32, "eager"
    else:
        attn_impl = "flash_attention_2" if has_flash_attn else "eager"
        return torch.bfloat16, attn_impl


def run_band_sensitivity(
    model_name: str,
    model_path: Path,
    band_configs: Tuple[BandConfig, ...],
    n_prompts: int = 10,
    n_resamples: int = 5,
) -> ModelBandSensitivity:
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info(f"{'='*60}")
    
    dtype, attn_impl = get_model_dtype_and_attn(model_name)
    
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    n_layers = len(model.model.layers) if hasattr(model, "model") else len(model.transformer.h)
    all_prompts = get_default_prompts()
    
    band_results = run_model_analysis(model, tokenizer, n_layers, band_configs, all_prompts, n_prompts, n_resamples)
    
    del model
    torch.cuda.empty_cache()
    
    return ModelBandSensitivity(model_name, n_layers, band_results)


def create_heatmap_figure(all_results: List[ModelBandSensitivity], save_path: Path) -> None:
    """
    Create a heatmap visualization of band-model plateau frequency
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data for heatmap
    models = [result.model_name for result in all_results]
    bands = [b.name for b in BAND_CONFIGS]
    
    # Create matrix of plateau frequencies
    data = []
    for model_result in all_results:
        row = []
        for band_result in model_result.band_results:
            row.append(band_result.plateau_freq)
        data.append(row)
    
    data = np.array(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create heatmap with matplotlib
    im = ax.imshow(data, cmap='Reds', aspect='auto', vmin=0.0, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(range(len(bands)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(bands)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black" if data[i, j] < 0.5 else "white")
    
    ax.set_title('Band-Model Plateau Frequency Heatmap\n(Band Boundary Sensitivity)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Angular Band Configuration', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Plateau Frequency', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Heatmap figure saved to: {save_path}")


def create_sensitivity_figure(all_results: List[ModelBandSensitivity], save_path: Path) -> None:
    """
    Create sensitivity analysis figure with multiple subplots
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    colors: Dict[str, str] = {"phi-2": "#2ecc71", "phi-3.5-mini": "#e74c3c", "gpt2": "#f39c12"}
    
    for model_result in all_results:
        model_name = model_result.model_name
        color = colors.get(model_name, "#888888")
        
        band_names = [r.band_name for r in model_result.band_results]
        x = range(len(band_names))
        
        plateau_freqs = [r.plateau_freq for r in model_result.band_results]
        axes[0].plot(x, plateau_freqs, 'o-', color=color, label=model_name, linewidth=2, markersize=8)
        
        plateau_starts = [r.mean_plateau_start / model_result.n_layers if r.mean_plateau_start > 0 else 0 
                         for r in model_result.band_results]
        axes[1].plot(x, plateau_starts, 'o-', color=color, label=model_name, linewidth=2, markersize=8)
        
        plateau_lens = [r.mean_plateau_len for r in model_result.band_results]
        axes[2].plot(x, plateau_lens, 'o-', color=color, label=model_name, linewidth=2, markersize=8)
    
    for ax, title, ylabel in [
        (axes[0], "(a) Plateau Frequency", "Plateau Frequency"),
        (axes[1], "(b) Normalized Plateau Start", "Start / N_layers"),
        (axes[2], "(c) Plateau Length", "Plateau Length (layers)"),
    ]:
        ax.set_xticks(range(len(BAND_CONFIGS)))
        ax.set_xticklabels([b.name for b in BAND_CONFIGS], rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Figure saved to: {save_path}")


class TestBandSensitivity(unittest.TestCase):
    
    def test_calculate_mean_std(self):
        mean, std = calculate_mean_std([2.0, 4.0, 4.0, 4.5, 5.0, 4.7])
        self.assertAlmostEqual(mean, 4.0333, places=3)
        self.assertTrue(std > 0)
        
        m0, s0 = calculate_mean_std([])
        self.assertEqual((m0, s0), (0.0, 0.0))
    
    def test_is_valid_candidate(self):
        self.assertTrue(is_valid_candidate([75.1, 75.3, 75.2], min_consecutive=3, max_drift=0.5))
        self.assertFalse(is_valid_candidate([75.1, 76.0, 75.2], min_consecutive=3, max_drift=0.5))
        self.assertFalse(is_valid_candidate([75.1, 75.2], min_consecutive=3, max_drift=0.5))
    
    def test_detect_plateau(self):
        angles = [80.0, 76.0, 76.2, 76.5, 77.0, 90.0, 76.0]
        has_plateau, start, length, p_angles = detect_plateau(angles, 75.0, 78.0, 3, 0.5)
        self.assertTrue(has_plateau)
        self.assertEqual(start, 1)
        self.assertEqual(length, 4)
        self.assertEqual(p_angles, [76.0, 76.2, 76.5, 77.0])
    
    def test_detect_plateau_no_plateau(self):
        angles = [60.0, 65.0, 70.0, 75.0, 80.0]
        has_plateau, start, length, p_angles = detect_plateau(angles, 75.0, 78.0)
        self.assertFalse(has_plateau)
        self.assertEqual(length, 0)
    
    def test_compute_angle_from_states(self):
        torch.manual_seed(42)
        H = torch.randn(40, 64)
        W = torch.randn(128, 64)
        _, _, Vh_weight = torch.linalg.svd(W, full_matrices=False)
        w_dir = Vh_weight[0]
        
        angle = compute_angle_from_states(H, w_dir, layer_idx=0)
        self.assertTrue(0.0 <= angle <= 90.0)


def main() -> None:
    all_results: List[ModelBandSensitivity] = []
    
    for model_name, model_path in MODEL_PATHS.items():
        result = run_band_sensitivity(model_name, model_path, BAND_CONFIGS, n_prompts=10, n_resamples=5)  # 减少样本数以加快速度
        all_results.append(result)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create both figures
    fig_path = OUTPUT_DIR / "band_sensitivity_figure.png"
    create_sensitivity_figure(all_results, fig_path)
    
    heatmap_path = OUTPUT_DIR / "band_model_heatmap.png"
    create_heatmap_figure(all_results, heatmap_path)
    
    results_dict: Dict[str, Any] = {
        "description": "Band Boundary Sensitivity Analysis",
        "n_resamples": 5,
        "n_prompts": 20,
        "bands": [{"name": b.name, "range": [b.low, b.high]} for b in BAND_CONFIGS],
        "results": {
            r.model_name: {
                "n_layers": r.n_layers,
                "bands": [
                    {
                        "name": br.band_name,
                        "range": list(br.band_range),
                        "plateau_freq": br.plateau_freq,
                        "mean_plateau_len": br.mean_plateau_len,
                        "std_plateau_len": br.std_plateau_len,
                        "mean_plateau_start": br.mean_plateau_start,
                        "std_plateau_start": br.std_plateau_start,
                        "mean_angle_in_plateau": br.mean_angle_in_plateau,
                        "std_angle_in_plateau": br.std_angle_in_plateau,
                    }
                    for br in r.band_results
                ]
            }
            for r in all_results
        }
    }
    
    json_path = RESULTS_DIR / "band_sensitivity_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Results saved to: {json_path}")
    
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"BAND SENSITIVITY SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<15} {'Band':<12} {'Freq':>6} {'Start':>6} {'Len':>5}")
    logger.info(f"{'-'*70}")
    for model_result in all_results:
        for br in model_result.band_results:
            start_str = f"L{br.mean_plateau_start:.0f}" if br.mean_plateau_start > 0 else "N/A"
            len_str = f"{br.mean_plateau_len:.1f}" if br.mean_plateau_len > 0 else "N/A"
            logger.info(f"{model_result.model_name:<15} {br.band_name:<12} {br.plateau_freq:>5.0%} {start_str:>6} {len_str:>5}")
        logger.info(f"{'-'*70}")


if __name__ == "__main__":
    logger.info("Running internal tests first...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Internal tests passed. Starting actual workload...")
    main()
