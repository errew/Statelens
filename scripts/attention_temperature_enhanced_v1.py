# -*- coding: utf-8 -*-
"""
Enhanced Attention Temperature Scaling Experiment
Test: τ = τ_min + C/β

Usage:
    Run experiment: python script.py
    Run unit tests: python script.py --test
"""

import gc
import json
import logging
import math
import sys
import time
import unittest
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import linregress, t as t_dist
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# ============================================================================
# Configurations & Constants
# ============================================================================

logging.basicConfig(format="[%(levelname)s] [%(asctime)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("EnhancedAttnTemp")

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
MODEL_BASE: Final[Path] = Path("E:/ComfyUI_windows_portable/ComfyUI/models/checkpoints")
OUTPUT_DIR: Final[Path] = BASE_DIR / "validation_results" / "attention_temperature_enhanced"

K_VALUES: Final[Tuple[int, ...]] = (1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100)
TEMPERATURES: Final[Tuple[float, ...]] = (0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)

MODELS: Final[Tuple[str, ...]] = (
    "phi-2",          # MHA + RoPE (efficient MHA)
    "gemma-2b",       # MQA + RoPE (multi-query attention)
    "Qwen2.5-1.5B",   # GQA + RoPE (validated)
)

MODEL_DEPTHS: Final[Dict[str, int]] = {
    "gpt2": 12,
    "phi-2": 32,
    "gemma-2b": 18,
    "Qwen2.5-0.5B": 24,
    "Qwen2.5-1.5B": 28,
    "Qwen2.5-3B": 36,
}

MODEL_ARCH_INFO: Final[Dict[str, Dict[str, str]]] = {
    "gpt2": {"attention": "MHA", "position": "Learned"},
    "phi-2": {"attention": "MHA", "position": "RoPE"},
    "gemma-2b": {"attention": "MQA", "position": "RoPE"},
    "Qwen2.5-0.5B": {"attention": "GQA", "position": "RoPE"},
    "Qwen2.5-1.5B": {"attention": "GQA", "position": "RoPE"},
    "Qwen2.5-3B": {"attention": "GQA", "position": "RoPE"},
}

PROMPTS: Final[Tuple[str, ...]] = (
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "Scientists have discovered a new species of deep-sea fish.",
    "The ancient city was hidden beneath layers of sediment.",
    "Music has the power to transcend cultural boundaries.",
    "The quantum computer solved the problem in milliseconds.",
    "She walked through the forest, listening to bird songs.",
    "The recipe calls for fresh herbs and olive oil.",
    "Climate change is affecting ecosystems worldwide.",
    "The artist painted a masterpiece over several months.",
)

EPS: Final[float] = 1e-8


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class TauFitResult:
    tau: float
    c1: float
    r_squared: float


@dataclass(frozen=True)
class CrossModelAnalysis:
    n_models: int
    depths: List[int]
    tau_mins: List[float]
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    interpretation: str


@dataclass(frozen=True)
class LogLogAnalysis:
    slope: float
    slope_stderr: float
    slope_ci: Tuple[float, float]
    contains_minus_one: bool


@dataclass
class ModelAnalysisResult:
    fit_method: str = "failed"
    tau_min: float = 0.0
    tau_min_stderr: float = 0.0
    C: float = 0.0
    C_stderr: float = 0.0
    r_squared: float = 0.0
    n_points: int = 0
    predicted_tau: List[float] = field(default_factory=list)
    log_log_analysis: Optional[LogLogAnalysis] = None


# ============================================================================
# Pure Mathematical & Statistical Functions
# ============================================================================

def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the coefficient of determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > EPS else 0.0


def compute_angle(direction: np.ndarray, pca_components: np.ndarray) -> float:
    """Computes the angle in degrees between a vector and a subspace."""
    if pca_components.size == 0 or np.linalg.norm(direction) < EPS:
        return 90.0
    
    dir_norm = direction / (np.linalg.norm(direction) + EPS)
    comp_norms = np.linalg.norm(pca_components, axis=1, keepdims=True) + EPS
    normalized_comps = pca_components / comp_norms
    
    projections = np.dot(normalized_comps, dir_norm)
    proj_norm_sq = np.sum(projections ** 2)
    
    proj_norm = math.sqrt(max(0.0, float(proj_norm_sq)))
    # Clip to [-1.0, 1.0] to prevent math.acos domain errors from floating point imprecision
    return math.degrees(math.acos(max(-1.0, min(1.0, proj_norm))))


def alignment_saturation(k: np.ndarray, c0: float, c1: float, tau: float) -> np.ndarray:
    """Saturation curve mapping PCA dimension k to cosine similarity."""
    return c0 + c1 * (1.0 - np.exp(-k / (tau + EPS)))


def tau_model(beta: np.ndarray, tau_min: float, c_param: float) -> np.ndarray:
    """Mathematical model for tau scaling: τ = τ_min + C/β."""
    return tau_min + c_param / (beta + EPS)


def fit_tau_alignment(angles: List[float], k_values: Tuple[int, ...]) -> TauFitResult:
    """Fits the alignment saturation curve to compute characteristic length Tau."""
    if len(angles) < 3:
        return TauFitResult(0.0, 0.0, 0.0)

    theta_array = np.array(angles, dtype=np.float64)
    k_array = np.array(k_values[:len(angles)], dtype=np.float64)
    
    cos_theta = np.cos(np.radians(theta_array))
    cos_min, cos_max = float(np.min(cos_theta)), float(np.max(cos_theta))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                alignment_saturation, k_array, cos_theta,
                p0=[cos_min, cos_max - cos_min, 30.0],
                bounds=([0.0, 0.0, 1.0],[1.0, 1.0, 500.0]),
                maxfev=5000
            )
        c0, c1, tau = popt
        cos_pred = alignment_saturation(k_array, float(c0), float(c1), float(tau))
        r2 = compute_r_squared(cos_theta, cos_pred)
        return TauFitResult(float(tau), float(c1), float(r2))
        
    except (RuntimeError, ValueError, OptimizeWarning):
        return TauFitResult(0.0, 0.0, 0.0)


# ============================================================================
# Core Engine Classes
# ============================================================================

class OnlineCovariance:
    """Welford's online algorithm for computing high-dimensional covariance."""
    
    def __init__(self, dim: int) -> None:
        self.dim: int = dim
        self.n: int = 0
        self.mean: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.M2: np.ndarray = np.zeros((dim, dim), dtype=np.float64)

    def add_batch(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"Batch must be 2D with shape (m, {self.dim})")
            
        m = batch.shape[0]
        if m == 0:
            return

        batch_mean = np.mean(batch, axis=0)
        batch_M2 = np.dot((batch - batch_mean).T, (batch - batch_mean))

        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = m
        else:
            n_new = self.n + m
            delta = batch_mean - self.mean
            
            self.mean += delta * (m / n_new)
            self.M2 += batch_M2 + np.outer(delta, delta) * (self.n * m / n_new)
            self.n = n_new

    def get_cov(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.M2)
        return self.M2 / (self.n - 1.0)


class AttentionScaleContext:
    """Context manager to cleanly modify attention temperature scaling."""
    
    def __init__(self, model: PreTrainedModel, temperature_scale: float) -> None:
        self.model = model
        self.scale_factor = temperature_scale
        self._original_scales: Dict[int, float] = {}

    def __enter__(self) -> "AttentionScaleContext":
        n_hooks = 0
        for layer in get_layers(self.model):
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if hasattr(attn, "scaling"):
                    original = float(attn.scaling)
                    self._original_scales[id(attn)] = original
                    modified = original * self.scale_factor
                    attn.scaling = modified
                    n_hooks += 1
                    if n_hooks == 1:
                        logger.debug(f"Layer 0 scaling patched: {original:.6f} → {modified:.6f}")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        for layer in get_layers(self.model):
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if id(attn) in self._original_scales:
                    attn.scaling = self._original_scales[id(attn)]
        self._original_scales.clear()


# ============================================================================
# PyTorch & Model Utilities
# ============================================================================

def cleanup_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Safely extract transformer layers regardless of specific model architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return nn.ModuleList()


def get_mlp_weight(layer: nn.Module) -> Optional[np.ndarray]:
    """Extract and compute the effective MLP weight matrix."""
    def get_w(name: str) -> Optional[torch.Tensor]:
        for w in [layer] + list(layer.children()):
            if hasattr(w, name):
                attr = getattr(w, name)
                if hasattr(attr, "weight"):
                    return attr.weight.detach().float().cpu()
        return None

    # LLaMA / Qwen / Gemma with merged gate_up_proj
    gate_up, down = get_w("gate_up_proj"), get_w("down_proj")
    if gate_up is not None and down is not None:
        return (down @ gate_up[: down.shape[1], :]).numpy()

    # LLaMA / Qwen with separate gate_proj and up_proj
    gate, up, down = get_w("gate_proj"), get_w("up_proj"), get_w("down_proj")
    if gate is not None and up is not None and down is not None:
        return (down @ gate).numpy()

    # phi-2 / GPT-Neo style: fc1 and fc2 (standard Linear)
    fc1, fc2 = get_w("fc1"), get_w("fc2")
    if fc1 is not None and fc2 is not None:
        return (fc2 @ fc1).numpy()

    # GPT-2 MLP pattern: c_fc and c_proj (Conv1D - weight shape is transposed)
    # Conv1D: weight shape = (in_features, out_features)
    # c_fc: (768, 3072), c_proj: (3072, 768)
    # For MLP direction: W = c_proj.weight.T @ c_fc.weight
    c_fc, c_proj = get_w("c_fc"), get_w("c_proj")
    if c_fc is not None and c_proj is not None:
        return (c_proj.T @ c_fc).numpy()

    return None


# ============================================================================
# Extractor Functions
# ============================================================================

def _extract_mlp_directions(layers: nn.ModuleList) -> List[Optional[np.ndarray]]:
    directions: List[Optional[np.ndarray]] =[]
    for layer in layers:
        w_mlp = get_mlp_weight(layer)
        if w_mlp is not None:
            try:
                u_mat, _, _ = np.linalg.svd(w_mlp, full_matrices=False)
                directions.append(u_mat[:, 0])
            except np.linalg.LinAlgError:
                directions.append(None)
        else:
            directions.append(None)
    return directions


def _collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layers: nn.ModuleList,
    hidden_dim: int,
    prompts: Tuple[str, ...],
    batch_size: int,
) -> List[OnlineCovariance]:
    cov_calculators = [OnlineCovariance(hidden_dim) for _ in layers]
    handles = []

    def make_hook(idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inp: Any, out: Any) -> None:
            tensor_out = out[0] if isinstance(out, tuple) else out
            activations = tensor_out.detach().float().cpu().numpy().reshape(-1, hidden_dim)
            cov_calculators[idx].add_batch(activations)
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        for i in range(0, len(prompts), batch_size):
            inputs = tokenizer(
                list(prompts[i:i + batch_size]),
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return cov_calculators


def measure_attention_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Tuple[str, ...],
    batch_size: int,
    n_samples: int = 10,
) -> Dict[str, float]:
    """
    Measure attention entropy directly via tensor operations (Vectorized).
    H = -Σ A_ij log(A_ij)
    """
    entropies: List[float] =[]
    handles =[]

    def entropy_hook(_module: nn.Module, _args: Any, output: Any) -> None:
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None and attn_weights.dim() == 4:
                # Vectorized Entropy Calculation on GPU
                probs = torch.clamp(attn_weights, min=1e-10)
                h_entropy = -torch.sum(probs * torch.log(probs), dim=-1)
                
                # Take mean over heads and sequence, keep batch dimension
                batch_entropies = h_entropy.mean(dim=(1, 2)).detach().float().cpu().numpy()
                entropies.extend(batch_entropies.tolist())

    for layer in get_layers(model):
        if hasattr(layer, "self_attn"):
            handles.append(layer.self_attn.register_forward_hook(entropy_hook))

    try:
        sample_prompts = prompts[:n_samples]
        for i in range(0, len(sample_prompts), batch_size):
            inputs = tokenizer(
                list(sample_prompts[i:i + batch_size]),
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(model.device)
            with torch.no_grad():
                model(**inputs, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "std_entropy": float(np.std(entropies)) if entropies else 0.0,
        "n_samples": len(entropies),
    }


def measure_tau_alignment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    hidden_dim: int,
    prompts: Tuple[str, ...],
    batch_size: int,
) -> Dict[str, Any]:
    """Combines execution of SVD, covariance tracking, and alignment fitting."""
    layers = get_layers(model)
    mlp_directions = _extract_mlp_directions(layers)
    cov_calculators = _collect_activations(model, tokenizer, layers, hidden_dim, prompts, batch_size)
    
    layer_results: Dict[str, Any] = {}
    valid_taus: List[float] =[]

    for idx, (cov_calc, direction) in enumerate(zip(cov_calculators, mlp_directions)):
        if direction is None:
            continue
            
        cov = cov_calc.get_cov()
        if cov.shape[0] < 2:
            continue
            
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_idx]
        except np.linalg.LinAlgError:
            continue

        angles =[]
        for k in K_VALUES:
            if k > eigenvectors.shape[1]:
                break
            pca_components = eigenvectors[:, :k].T
            angles.append(compute_angle(direction, pca_components))

        fit_result = fit_tau_alignment(angles, K_VALUES)
        
        if fit_result.tau > 0 and fit_result.r_squared > 0.9:
            layer_results[str(idx)] = {
                "tau": fit_result.tau,
                "r_squared": fit_result.r_squared,
                "angles": angles
            }
            valid_taus.append(fit_result.tau)

    return {
        "layer_results": layer_results,
        "tau_mean": float(np.mean(valid_taus)) if valid_taus else 0.0,
        "tau_std": float(np.std(valid_taus)) if valid_taus else 0.0,
        "n_valid_layers": len(valid_taus),
    }


# ============================================================================
# Statistical Analysis Module
# ============================================================================

def analyze_temperature_scaling(
    beta_values: np.ndarray,
    tau_values: np.ndarray,
    confidence: float = 0.95
) -> ModelAnalysisResult:
    """Rigorous temperature scaling curve fit: τ = τ_min + C/β."""
    result = ModelAnalysisResult(n_points=len(beta_values))
    
    if len(beta_values) < 2:
        return result

    try:
        popt, pcov = curve_fit(
            tau_model, beta_values, tau_values,
            p0=[30.0, 10.0],
            bounds=([1.0, 0.1], [500.0, 1000.0]),
            maxfev=10000,
        )
        tau_min_fit, c_fit = popt
        tau_predicted = tau_model(beta_values, float(tau_min_fit), float(c_fit))
        r_squared = compute_r_squared(tau_values, tau_predicted)
        
        perr = np.sqrt(np.diag(pcov))
        
        result.fit_method = "curve_fit"
        result.tau_min = float(tau_min_fit)
        result.C = float(c_fit)
        result.tau_min_stderr = float(perr[0])
        result.C_stderr = float(perr[1])
        result.r_squared = float(r_squared)
        result.predicted_tau = tau_predicted.tolist()

    except (RuntimeError, ValueError, OptimizeWarning):
        # Fallback to linear regression on 1/β
        beta_inv = 1.0 / (beta_values + EPS)
        lr_res = linregress(beta_inv, tau_values)
        
        result.fit_method = "linear_regression"
        result.tau_min = float(lr_res.intercept)
        result.C = float(lr_res.slope)
        result.tau_min_stderr = float(lr_res.intercept_stderr)
        result.C_stderr = float(lr_res.stderr)
        result.r_squared = float(lr_res.rvalue ** 2)
        result.predicted_tau = tau_model(beta_values, result.tau_min, result.C).tolist()

    # Log-Log Slope Analysis for baseline comparison
    tau_min_naive = float(np.min(tau_values))
    tau_corrected = tau_values - tau_min_naive
    non_zero_mask = tau_corrected > 0.1
    
    if np.sum(non_zero_mask) >= 2:
        log_beta = np.log10(beta_values[non_zero_mask])
        log_tau_corrected = np.log10(tau_corrected[non_zero_mask])
        
        ll_res = linregress(log_beta, log_tau_corrected)
        df = np.sum(non_zero_mask) - 2
        t_crit = t_dist.ppf((1 + confidence) / 2, df)
        
        ci_lower = float(ll_res.slope - t_crit * ll_res.stderr)
        ci_upper = float(ll_res.slope + t_crit * ll_res.stderr)
        
        result.log_log_analysis = LogLogAnalysis(
            slope=float(ll_res.slope),
            slope_stderr=float(ll_res.stderr),
            slope_ci=(ci_lower, ci_upper),
            contains_minus_one=(ci_lower <= -1.0 <= ci_upper)
        )

    return result


def perform_cross_model_analysis(tau_mins: List[float], depths: List[int]) -> Optional[CrossModelAnalysis]:
    """Performs statistical analysis across different model depths."""
    if len(tau_mins) < 2:
        return None
        
    depth_array = np.array(depths, dtype=np.float64)
    tau_min_array = np.array(tau_mins, dtype=np.float64)
    
    res = linregress(depth_array, tau_min_array)
    return CrossModelAnalysis(
        n_models=len(tau_mins),
        depths=depths,
        tau_mins=tau_mins,
        slope=float(res.slope),
        intercept=float(res.intercept),
        r_squared=float(res.rvalue ** 2),
        p_value=float(res.pvalue),
        interpretation="τ_min ∝ L" if res.pvalue < 0.05 else "No significant relationship",
    )


# ============================================================================
# Main Experiment Flow
# ============================================================================

def run_single_model(
    model_name: str,
    temperatures: Tuple[float, ...],
    n_prompts: int,
    batch_size: int,
) -> Dict[str, Any]:
    """End-to-end evaluation pipeline for a single model."""
    arch_info = MODEL_ARCH_INFO.get(model_name, {})
    logger.info(f"\n{'='*70}\nMODEL: {model_name} (L={MODEL_DEPTHS.get(model_name, '?')}, Arch: {arch_info.get('attention', '?')})\n{'='*70}")
    
    model_path = MODEL_BASE / model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set pad_token for models that don't have one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    
    hidden_dim = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 768))
    extended_prompts = (PROMPTS * (n_prompts // len(PROMPTS) + 1))[:n_prompts]
    
    results: Dict[str, Any] = {
        "model": model_name,
        "depth": MODEL_DEPTHS.get(model_name, None),
        "hidden_dim": hidden_dim,
        "n_prompts": n_prompts,
        "architecture": arch_info,
        "temperatures": {},
    }
    
    for temp in temperatures:
        logger.info(f"--- Temperature Scale: {temp} ---")
        
        with AttentionScaleContext(model, temp):
            tau_results = measure_tau_alignment(model, tokenizer, hidden_dim, extended_prompts, batch_size)
            entropy_results = measure_attention_entropy(model, tokenizer, extended_prompts, batch_size)
            
            tau_results["entropy"] = entropy_results
            results["temperatures"][str(temp)] = tau_results
            
            logger.info(f"τ_mean = {tau_results['tau_mean']:.2f} ± {tau_results['tau_std']:.2f}")
            logger.info(f"Entropy = {entropy_results['mean_entropy']:.4f}")
        
        cleanup_memory()
    
    beta_values = np.array([float(t) for t in temperatures])
    tau_values = np.array([results["temperatures"][str(t)]["tau_mean"] for t in temperatures])
    
    analysis = analyze_temperature_scaling(beta_values, tau_values)
    
    # Store results dynamically into dict (to be JSON serialized)
    results["analysis"] = analysis.__dict__.copy()
    if analysis.log_log_analysis:
        results["analysis"]["log_log_analysis"] = analysis.log_log_analysis.__dict__
    
    logger.info(f"--- Analysis for {model_name} ---")
    logger.info(f"Fit method: {analysis.fit_method} | R²: {analysis.r_squared:.4f}")
    logger.info(f"τ_min = {analysis.tau_min:.2f} ± {analysis.tau_min_stderr:.2f}")
    
    model.to("cpu")
    del model
    cleanup_memory()
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results: Dict[str, Any] = {
        "experiment": "attention_temperature_enhanced",
        "temperatures": list(TEMPERATURES),
        "models": {},
    }
    
    for model_name in MODELS:
        try:
            model_results = run_single_model(model_name, TEMPERATURES, n_prompts=50, batch_size=4)
            all_results["models"][model_name] = model_results
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to run {model_name} due to execution error: {e}")
            cleanup_memory()
            continue

    tau_mins, depths = [],[]
    for model_name, res in all_results["models"].items():
        if "analysis" in res and "tau_min" in res["analysis"]:
            tau_mins.append(res["analysis"]["tau_min"])
            depths.append(res.get("depth", MODEL_DEPTHS.get(model_name, 0)))

    cross_res = perform_cross_model_analysis(tau_mins, depths)
    if cross_res:
        all_results["cross_model_analysis"] = cross_res.__dict__
        logger.info(f"\n{'='*70}\nCROSS-MODEL ANALYSIS: τ_min vs Depth\n{'='*70}")
        logger.info(f"Slope: {cross_res.slope:.4f} | R²: {cross_res.r_squared:.4f} | p: {cross_res.p_value:.4f}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"enhanced_temperature_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults successfully saved to: {output_file}")


# ============================================================================
# Unit Tests (Run via `--test`)
# ============================================================================

class TestEnhancedAttnTemp(unittest.TestCase):
    def test_compute_angle_orthogonal(self) -> None:
        direction = np.array([1.0, 0.0, 0.0])
        pca = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        angle = compute_angle(direction, pca)
        self.assertAlmostEqual(angle, 90.0, places=4)

    def test_compute_angle_parallel(self) -> None:
        direction = np.array([1.0, 0.0, 0.0])
        pca = np.array([[1.0, 0.0, 0.0]])
        angle = compute_angle(direction, pca)
        self.assertAlmostEqual(angle, 0.0, places=4)

    def test_online_covariance(self) -> None:
        cov_calc = OnlineCovariance(2)
        batch1 = np.array([[1.0, 2.0],[3.0, 4.0]])
        batch2 = np.array([[5.0, 6.0],[7.0, 8.0]])
        cov_calc.add_batch(batch1)
        cov_calc.add_batch(batch2)
        
        expected_cov = np.cov(np.vstack((batch1, batch2)), rowvar=False)
        np.testing.assert_almost_equal(cov_calc.get_cov(), expected_cov, decimal=5)

    def test_alignment_saturation(self) -> None:
        k = np.array([10, 20, 30])
        c0, c1, tau = 0.0, 1.0, 10.0
        val = alignment_saturation(k, c0, c1, tau)
        self.assertAlmostEqual(val[0], 1.0 - math.exp(-1.0), places=4)

    def test_compute_r_squared(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.0])
        r2 = compute_r_squared(y_true, y_pred)
        self.assertTrue(0.9 < r2 <= 1.0)


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main()
    else:
        main()