"""
τ Profile Likelihood Analysis

核心问题：τ 是真实参数还是 truncation artifact？

方法：
1. 固定不同的 θ_min 值
2. 每次重新拟合 A 和 τ
3. 画 τ vs θ_min 曲线

判据：
- τ 基本不变 → τ 是真实参数
- τ 随 θ_min 明显变化 → τ 稳定性是假象
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "validation_results" / "statlens_v4" / "multi_arch_results.json"
OUTPUT_DIR = BASE_DIR / "validation_results" / "profile_likelihood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def exp_decay(K, theta_min, A, tau):
    return theta_min + A * np.exp(-K / tau)


def fit_with_fixed_theta_min(K_values, thetas, theta_min_fixed):
    K_arr = np.array(K_values)
    theta_arr = np.array(thetas)
    
    A_init = theta_arr[0] - theta_min_fixed
    if A_init <= 0:
        A_init = 1.0
    
    def model(K, A, tau):
        return theta_min_fixed + A * np.exp(-K / tau)
    
    try:
        popt, _ = curve_fit(
            model, K_arr, theta_arr,
            p0=[A_init, 50],
            bounds=([0.1, 1], [100, 500]),
            maxfev=5000
        )
        A_fit, tau_fit = popt
        
        y_pred = model(K_arr, A_fit, tau_fit)
        residuals = theta_arr - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((theta_arr - np.mean(theta_arr))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return A_fit, tau_fit, r2
    except:
        return None, None, -1


def analyze_layer(layer_data, theta_min_range=None):
    angles = layer_data["angles"]
    K_values = sorted([int(k) for k in angles.keys()])
    thetas = [angles[str(k)] for k in K_values]
    
    theta_actual_min = min(thetas)
    theta_max = max(thetas)
    
    if theta_min_range is None:
        theta_min_range = np.linspace(
            max(0, theta_actual_min - 20),
            min(90, theta_max - 5),
            21
        )
    
    results = []
    for theta_min_fixed in theta_min_range:
        if theta_min_fixed >= theta_max:
            continue
        A_fit, tau_fit, r2 = fit_with_fixed_theta_min(K_values, thetas, theta_min_fixed)
        if tau_fit is not None:
            results.append({
                'theta_min_fixed': theta_min_fixed,
                'A_fit': A_fit,
                'tau_fit': tau_fit,
                'r2': r2
            })
    
    return results, theta_actual_min


def compute_tau_stability_metric(profile_results):
    if len(profile_results) < 3:
        return None, None
    
    taus = [r['tau_fit'] for r in profile_results]
    theta_mins = [r['theta_min_fixed'] for r in profile_results]
    
    tau_mean = np.mean(taus)
    tau_std = np.std(taus)
    cv = tau_std / tau_mean if tau_mean > 0 else float('inf')
    
    correlation = np.corrcoef(theta_mins, taus)[0, 1]
    
    return cv, correlation


def main():
    print("=" * 60)
    print("τ Profile Likelihood Analysis")
    print("=" * 60)
    
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    
    all_stability_metrics = []
    
    for model_name, model_data in data.items():
        print(f"\n[INFO] Processing: {model_name}")
        
        layer_results = model_data.get("layer_results", {})
        
        model_profile_data = []
        
        for layer_idx_str, layer_data in layer_results.items():
            layer_idx = int(layer_idx_str)
            
            profile_results, theta_actual_min = analyze_layer(layer_data)
            
            if len(profile_results) < 3:
                continue
            
            cv, correlation = compute_tau_stability_metric(profile_results)
            
            if cv is not None:
                all_stability_metrics.append({
                    'model': model_name,
                    'layer': layer_idx,
                    'cv': cv,
                    'correlation': correlation,
                    'theta_actual_min': theta_actual_min
                })
            
            model_profile_data.append({
                'layer_idx': layer_idx,
                'profile_results': profile_results,
                'theta_actual_min': theta_actual_min,
                'cv': cv,
                'correlation': correlation
            })
        
        if model_profile_data:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            n_layers = len(model_profile_data)
            colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
            
            ax1 = axes[0, 0]
            for i, lp in enumerate(model_profile_data):
                theta_mins = [r['theta_min_fixed'] for r in lp['profile_results']]
                taus = [r['tau_fit'] for r in lp['profile_results']]
                ax1.plot(theta_mins, taus, 'o-', color=colors[i], alpha=0.7,
                        label=f"L{lp['layer_idx']}" if i % max(1, n_layers//6) == 0 else "")
            ax1.set_xlabel('Fixed θ_min (degrees)')
            ax1.set_ylabel('Fitted τ')
            ax1.set_title(f'{model_name}: τ vs θ_min Profile')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[0, 1]
            cvs = [lp['cv'] for lp in model_profile_data if lp['cv'] is not None]
            layers = [lp['layer_idx'] for lp in model_profile_data if lp['cv'] is not None]
            ax2.bar(layers, cvs, color='steelblue', alpha=0.7)
            ax2.axhline(y=0.2, color='r', linestyle='--', label='CV=0.2 threshold')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('CV(τ)')
            ax2.set_title('Coefficient of Variation of τ across θ_min')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3 = axes[1, 0]
            corrs = [lp['correlation'] for lp in model_profile_data if lp['correlation'] is not None]
            layers = [lp['layer_idx'] for lp in model_profile_data if lp['correlation'] is not None]
            colors_bar = ['green' if c < 0.3 else 'orange' if c < 0.6 else 'red' for c in corrs]
            ax3.bar(layers, corrs, color=colors_bar, alpha=0.7)
            ax3.axhline(y=0.3, color='r', linestyle='--', label='Corr=0.3 threshold')
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Correlation(θ_min, τ)')
            ax3.set_title('τ-θ_min Correlation (low = stable τ)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            theta_mins_actual = [lp['theta_actual_min'] for lp in model_profile_data]
            ax4.scatter(theta_mins_actual, corrs, c=corrs, cmap='RdYlGn_r', 
                       s=50, alpha=0.7, edgecolors='black')
            ax4.set_xlabel('Actual θ_min (degrees)')
            ax4.set_ylabel('Correlation(θ_min, τ)')
            ax4.set_title('θ_min vs τ Stability')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{model_name}_profile_likelihood.png", dpi=150)
            plt.close(fig)
    
    if all_stability_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cvs = [m['cv'] for m in all_stability_metrics]
        corrs = [m['correlation'] for m in all_stability_metrics]
        
        ax1 = axes[0]
        ax1.hist(cvs, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(cvs), color='r', linestyle='--', 
                   label=f'Mean={np.mean(cvs):.3f}')
        ax1.axvline(x=np.median(cvs), color='g', linestyle='--', 
                   label=f'Median={np.median(cvs):.3f}')
        ax1.set_xlabel('CV(τ) across θ_min values')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of τ Stability (CV)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.hist(corrs, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(corrs), color='r', linestyle='--', 
                   label=f'Mean={np.mean(corrs):.3f}')
        ax2.axvline(x=np.median(corrs), color='g', linestyle='--', 
                   label=f'Median={np.median(corrs):.3f}')
        ax2.axvline(x=0.3, color='black', linestyle=':', label='Threshold=0.3')
        ax2.set_xlabel('Correlation(θ_min, τ)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of τ-θ_min Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "summary_stability_distribution.png", dpi=150)
        plt.close(fig)
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    if all_stability_metrics:
        cvs = [m['cv'] for m in all_stability_metrics]
        corrs = [m['correlation'] for m in all_stability_metrics]
        
        print(f"\nCV(τ) Statistics:")
        print(f"  Mean:   {np.mean(cvs):.3f}")
        print(f"  Median: {np.median(cvs):.3f}")
        print(f"  Std:    {np.std(cvs):.3f}")
        print(f"  Range:  [{np.min(cvs):.3f}, {np.max(cvs):.3f}]")
        
        print(f"\nCorrelation(θ_min, τ) Statistics:")
        print(f"  Mean:   {np.mean(corrs):.3f}")
        print(f"  Median: {np.median(corrs):.3f}")
        print(f"  Std:    {np.std(corrs):.3f}")
        print(f"  Range:  [{np.min(corrs):.3f}, {np.max(corrs):.3f}]")
        
        low_corr_count = sum(1 for c in corrs if abs(c) < 0.3)
        total = len(corrs)
        print(f"\n  Layers with |correlation| < 0.3: {low_corr_count}/{total} ({100*low_corr_count/total:.1f}%)")
        
        mean_abs_corr = np.mean([abs(c) for c in corrs])
        median_abs_corr = np.median([abs(c) for c in corrs])
        
        print("\n" + "=" * 60)
        print("Interpretation")
        print("=" * 60)
        
        if mean_abs_corr < 0.3 and median_abs_corr < 0.3:
            print("\n✓ τ is STABLE across different θ_min assumptions")
            print("  → τ appears to be a REAL parameter, not an artifact")
        elif mean_abs_corr > 0.5:
            print("\n✗ τ VARIES strongly with θ_min")
            print("  → τ stability is likely an ILLUSION (parameter coupling)")
            print(f"  → Mean |correlation| = {mean_abs_corr:.3f}")
        else:
            print("\n~ τ shows MODERATE sensitivity to θ_min")
            print("  → Results are ambiguous, need further investigation")
    
    mean_abs_corr = np.mean([abs(c) for c in corrs]) if corrs else 0
    summary = {
        'n_layers_analyzed': len(all_stability_metrics),
        'cv_mean': float(np.mean(cvs)) if cvs else None,
        'cv_median': float(np.median(cvs)) if cvs else None,
        'correlation_mean': float(np.mean(corrs)) if corrs else None,
        'correlation_median': float(np.median(corrs)) if corrs else None,
        'abs_correlation_mean': float(mean_abs_corr),
        'low_correlation_fraction': low_corr_count / total if total > 0 else None,
        'interpretation': 'stable' if mean_abs_corr < 0.3 else 'unstable' if mean_abs_corr > 0.5 else 'ambiguous'
    }
    
    with open(OUTPUT_DIR / "profile_likelihood_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
