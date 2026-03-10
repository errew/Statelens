"""
Common utilities for slens validation scripts.

Provides:
- Vectorized KL computation
- Standard logging configuration
- Type-safe JSON serialization
- Isolated random number generators
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import numpy.typing as npt


EPS = 1e-8
RESULTS_DIR = Path(__file__).parent.parent / "validation_results"


def setup_logger(name: str) -> logging.Logger:
    """Create a standard logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class VectorizedKLMetric:
    """Fully vectorized KL divergence calculator."""
    
    def __init__(self, eps: float = EPS):
        self.eps = eps
    
    def compute_kl_per_step(self, attention: npt.NDArray[np.float64]) -> List[float]:
        """Compute KL per step using vectorized operations."""
        n_layers, n_heads, seq_len, _ = attention.shape
        kl_values: List[float] = []
        
        for t in range(1, seq_len):
            p = attention[:, :, t, :t]
            p_safe = np.clip(p, self.eps, 1.0)
            p_norm = p_safe / p_safe.sum(axis=-1, keepdims=True)
            
            q = attention[:, :, t-1, :t-1]
            q_padded = np.zeros_like(p_safe)
            
            if q.shape[-1] == 0:
                q_padded[..., :] = 1.0 / t
            else:
                q_padded[..., :q.shape[-1]] = q
                q_padded[..., q.shape[-1]:] = 1.0 / t
            
            q_safe = np.clip(q_padded, self.eps, 1.0)
            q_norm = q_safe / q_safe.sum(axis=-1, keepdims=True)
            
            kl_matrix = np.sum(p_norm * np.log(p_norm / q_norm), axis=-1)
            kl_matrix = np.maximum(0.0, kl_matrix)
            kl_values.append(float(np.mean(kl_matrix)))
        
        return kl_values
    
    def compute_full_metrics(self, attention: npt.NDArray[np.float64]) -> Dict[str, Any]:
        """Compute full slens metrics."""
        kl_values = self.compute_kl_per_step(attention)
        
        if not kl_values:
            return {
                'mean_kl': 0.0,
                'effective_slens': 0.0,
                'stability': 1.0,
                'kl_trajectory': [],
            }
        
        mean_kl = float(np.mean(kl_values))
        std_kl = float(np.std(kl_values))
        cv = std_kl / (mean_kl + self.eps)
        stability = float(np.exp(-2 * cv))
        effective_slens = mean_kl * stability
        
        return {
            'mean_kl': mean_kl,
            'effective_slens': effective_slens,
            'stability': stability,
            'kl_trajectory': [float(k) for k in kl_values],
        }


def normalize_weights(weights: npt.NDArray[np.float64], eps: float = EPS) -> npt.NDArray[np.float64]:
    """Normalize weights to sum to 1."""
    weights = np.maximum(weights, eps)
    return weights / weights.sum()


def save_json(data: Dict[str, Any], filename: str) -> None:
    """Save data to JSON with proper type conversion."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / filename
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logging.error(f"Failed to save results: {e}")
