"""
Stability Index (SI) Calculation for All Models
================================================

计算小模型和大模型的 SI。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS_DIR = Path(__file__).parent / "validation_results"
OUTPUT_PATH = Path(__file__).parent.parent / "v8" / "experiment_data" / "si_analysis_all_models.json"


def load_small_model_results() -> List[Dict]:
    results = []
    for json_file in sorted(RESULTS_DIR.glob("single_*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    return results


def load_large_model_results() -> List[Dict]:
    results = []
    for json_file in sorted(RESULTS_DIR.glob("large_model_*_corrected.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    return results


def find_longest_platform(angles: List[float], band: Tuple[float, float], tolerance: float = 2.0) -> int:
    lower, upper = band
    in_band = [lower - tolerance <= a <= upper + tolerance for a in angles]
    
    max_length = 0
    current_length = 0
    
    for is_in in in_band:
        if is_in:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    
    return max_length


def calculate_si(layer_results: List[Dict], n_layers: int, k: int = 100) -> Dict:
    angles_k100 = []
    for lr in layer_results:
        angle = lr['angles_min'].get(str(k), lr['angles_min'].get(k, float('nan')))
        angles_k100.append(angle)
    
    mean_angle = sum(angles_k100) / len(angles_k100)
    std_angle = math.sqrt(sum((a - mean_angle)**2 for a in angles_k100) / len(angles_k100))
    
    band_lower = mean_angle - std_angle
    band_upper = mean_angle + std_angle
    
    platform_length = find_longest_platform(angles_k100, (band_lower, band_upper), tolerance=std_angle)
    
    si = platform_length / n_layers
    
    return {
        "mean_angle": mean_angle,
        "std_angle": std_angle,
        "band": (band_lower, band_upper),
        "platform_length": platform_length,
        "si": si,
        "angles": angles_k100,
    }


def analyze_model(model_data: Dict, is_large: bool = False) -> Dict:
    n_layers = model_data['n_layers']
    layer_results = model_data['layer_results']
    
    si_result = calculate_si(layer_results, n_layers)
    
    angles = si_result['angles']
    angle_range = (min(angles), max(angles))
    angle_spread = angle_range[1] - angle_range[0]
    
    if si_result['si'] >= 0.5:
        stability_class = "High"
    elif si_result['si'] >= 0.25:
        stability_class = "Medium"
    else:
        stability_class = "Low"
    
    return {
        "model_name": model_data['model_name'],
        "architecture": model_data.get('architecture', 'unknown'),
        "n_layers": n_layers,
        "hidden_dim": model_data.get('hidden_dim', 0),
        "is_large": is_large,
        "mean_theta_min": si_result['mean_angle'],
        "std_theta_min": si_result['std_angle'],
        "angle_range": angle_range,
        "angle_spread": angle_spread,
        "band": si_result['band'],
        "platform_length": si_result['platform_length'],
        "si": si_result['si'],
        "stability_class": stability_class,
    }


def main():
    small_results = load_small_model_results()
    large_results = load_large_model_results()
    
    analyses = []
    
    print("=" * 80)
    print("SMALL MODELS (<4B)")
    print("=" * 80)
    for model_data in small_results:
        analysis = analyze_model(model_data, is_large=False)
        analyses.append(analysis)
        print(f"{analysis['model_name']:<20} SI={analysis['si']:.2f}  θ={analysis['mean_theta_min']:.1f}°±{analysis['std_theta_min']:.1f}°  {analysis['stability_class']}")
    
    print("\n" + "=" * 80)
    print("LARGE MODELS (6B-7B)")
    print("=" * 80)
    for model_data in large_results:
        analysis = analyze_model(model_data, is_large=True)
        analyses.append(analysis)
        print(f"{analysis['model_name']:<20} SI={analysis['si']:.2f}  θ={analysis['mean_theta_min']:.1f}°±{analysis['std_theta_min']:.1f}°  {analysis['stability_class']}")
    
    small_analyses = [a for a in analyses if not a['is_large']]
    large_analyses = [a for a in analyses if a['is_large']]
    
    output = {
        "experiment": "SI Analysis for All Models",
        "date": "2026-03-04",
        "method": "SI = Longest Platform Length / Total Layers",
        "band_definition": "mean_angle ± std_angle",
        "small_models": {a['model_name']: a for a in small_analyses},
        "large_models": {a['model_name']: a for a in large_analyses},
        "summary": {
            "small_models": {
                "count": len(small_analyses),
                "si_range": [min(a['si'] for a in small_analyses), max(a['si'] for a in small_analyses)],
                "theta_range": [min(a['mean_theta_min'] for a in small_analyses), max(a['mean_theta_min'] for a in small_analyses)],
            },
            "large_models": {
                "count": len(large_analyses),
                "si_range": [min(a['si'] for a in large_analyses), max(a['si'] for a in large_analyses)],
                "theta_range": [min(a['mean_theta_min'] for a in large_analyses), max(a['mean_theta_min'] for a in large_analyses)],
            },
        },
        "si_ranking": sorted(
            [{"model": a['model_name'], "si": a['si'], "theta": a['mean_theta_min'], "is_large": a['is_large']} for a in analyses],
            key=lambda x: x['si'],
            reverse=True
        ),
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 80)
    print("SI RANKING (All Models)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model':<20} {'Size':<8} {'SI':<8} {'θ_min':<10} {'Class'}")
    print("-" * 80)
    for i, a in enumerate(sorted(analyses, key=lambda x: x['si'], reverse=True), 1):
        size = "Large" if a['is_large'] else "Small"
        print(f"{i:<5} {a['model_name']:<20} {size:<8} {a['si']:<8.2f} {a['mean_theta_min']:.1f}°     {a['stability_class']}")


if __name__ == "__main__":
    main()
