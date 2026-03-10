"""
完整残差块 Jacobian 谱测试 - 决定性实验 (v2)
================================

目标：
- 测试完整残差块 J_block = ∂(h + Attention + MLP)/∂h 的谱特性
- 使用 Power Iteration 精确估计最大特征值
- 重点验证 Layer 2 在 High Redundancy 下的行为

方法：
- 使用完整模型前向传播 + 有限差分
- Power Iteration 锁定最大特征值
"""

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch.autograd.functional import jvp
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FullBlock")


@dataclass(frozen=True)
class Config:
    model_name: str = "Qwen2.5-0.5B"
    n_power_iter: int = 50
    n_samples: int = 100
    use_float64: bool = True


def load_model_from_comfyui(config: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, torch.dtype]:
    """从 ComfyUI 路径加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if config.use_float64 else torch.float32
    
    comfyui_path = Path(r"E:\ComfyUI_windows_portable\ComfyUI")
    model_path = comfyui_path / "models" / "checkpoints" / config.model_name
    
    if not model_path.exists():
        available_models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "gpt2"]
        for alt_model in available_models:
            alt_path = comfyui_path / "models" / "checkpoints" / alt_model
            if alt_path.exists():
                model_path = alt_path
                config = Config(model_name=alt_model)
                logger.info(f"使用替代模型: {alt_model}")
                break
        else:
            raise FileNotFoundError(f"未找到模型路径: {model_path}")
    
    logger.info(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto" if device == torch.device("cuda") else None,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    
    model.eval()
    return model, tokenizer, device, dtype


@contextmanager
def replace_hidden_state_hook(model, layer_idx, new_hidden):
    """Hook to replace hidden state at a specific layer"""
    storage = [None]
    
    def hook(module, args, kwargs):
        storage[0] = args[0] if args else kwargs.get('hidden_states')
        return None
    
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True)
    
    try:
        yield storage
    finally:
        handle.remove()


def compute_full_block_jvp_via_autograd(
    model: AutoModelForCausalLM,
    inputs: Dict,
    layer_idx: int,
    v: Tensor,
    device: torch.device,
    dtype: torch.dtype
) -> Optional[Tensor]:
    """
    使用 autograd.jvp 计算完整残差块的 Jacobian-vector product
    
    J_block @ v = ∂h_{n+1}/∂h_n @ v
    
    通过构造一个函数 f(h_n) -> h_{n+1}，然后计算 jvp
    """
    layer = model.model.layers[layer_idx]
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h_n = outputs.hidden_states[layer_idx][0, -1].clone().to(dtype)
        h_n_plus_1 = outputs.hidden_states[layer_idx + 1][0, -1].clone().to(dtype)
    
    del outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    def block_forward(h_vec: Tensor) -> Tensor:
        """
        完整残差块前向传播
        h_{n+1} = h_n + Attention(RMSNorm(h_n)) + MLP(RMSNorm(h_n + Attention))
        """
        h_input = h_vec.unsqueeze(0).unsqueeze(0)
        
        residual = h_input
        
        norm1 = layer.input_layernorm(residual)
        
        batch_size, seq_len = h_input.shape[:2]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        position_embeddings = model.model.rotary_emb(h_input, position_ids)
        
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=dtype)
        
        attn_out = layer.self_attn(
            hidden_states=norm1,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )[0]
        
        h_attn = residual + attn_out
        
        norm2 = layer.post_attention_layernorm(h_attn)
        mlp_out = layer.mlp(norm2)
        h_out = h_attn + mlp_out
        
        return h_out.squeeze()
    
    try:
        h_n_grad = h_n.detach().clone().requires_grad_(True)
        _, jvp_val = jvp(block_forward, (h_n_grad,), (v.to(dtype),), create_graph=False)
        return jvp_val
    except Exception as e:
        logger.warning(f"JVP 计算失败: {e}")
        return None


def power_iteration_max_eigenvalue(
    model: AutoModelForCausalLM,
    inputs: Dict,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    n_iter: int = 50
) -> Tuple[float, float]:
    """
    Power Iteration 估计完整残差块 Jacobian 的最大特征值
    
    J_block = I + J_Δh
    λ_max(J_block) = lim_{k->∞} ||J^k v|| / ||J^{k-1} v||
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h_n = outputs.hidden_states[layer_idx][0, -1].clone().to(dtype)
    
    del outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    dim = h_n.shape[0]
    v = torch.randn(dim, device=device, dtype=dtype)
    v = v / v.norm()
    
    logger.info(f"    Power Iteration (n_iter={n_iter})...")
    
    for i in range(n_iter):
        jvp_val = compute_full_block_jvp_via_autograd(
            model, inputs, layer_idx, v, device, dtype
        )
        
        if jvp_val is None:
            logger.warning(f"    迭代 {i} 失败，返回默认值")
            return 1.0, 0.0
        
        v = jvp_val
        norm_v = v.norm()
        
        if norm_v < 1e-10:
            logger.warning(f"    迭代 {i} 向量范数过小")
            break
        
        v = v / norm_v
        
        if (i + 1) % 10 == 0:
            logger.info(f"    迭代 {i+1}/{n_iter}, ||Jv|| = {norm_v:.6f}")
    
    final_jvp = compute_full_block_jvp_via_autograd(
        model, inputs, layer_idx, v, device, dtype
    )
    
    if final_jvp is None:
        return 1.0, 0.0
    
    lambda_max = torch.dot(v, final_jvp).item()
    
    rayleigh = lambda_max
    
    return lambda_max, rayleigh


def estimate_jacobian_spectrum_samples(
    model: AutoModelForCausalLM,
    inputs: Dict,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    n_samples: int = 100
) -> Tuple[Optional[Tensor], float]:
    """
    随机采样估计 Jacobian 谱分布
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h_n = outputs.hidden_states[layer_idx][0, -1].clone().to(dtype)
    
    del outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    dim = h_n.shape[0]
    J_samples = torch.zeros((n_samples, dim), dtype=dtype, device=device)
    
    logger.info(f"    随机采样 (n_samples={n_samples})...")
    
    for i in range(n_samples):
        v = torch.randn(dim, device=device, dtype=dtype)
        v = v / v.norm()
        
        jvp_val = compute_full_block_jvp_via_autograd(
            model, inputs, layer_idx, v, device, dtype
        )
        
        if jvp_val is not None:
            J_samples[i] = jvp_val
        
        if (i + 1) % 20 == 0:
            logger.info(f"    采样进度: {i+1}/{n_samples}")
    
    _, S, _ = torch.linalg.svd(J_samples, full_matrices=False)
    
    del J_samples
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return S, S[0].item() if len(S) > 0 else 0.0


def calculate_effective_rank(singular_values: Tensor) -> Tuple[float, float]:
    """计算有效秩"""
    s = singular_values.clone()
    p = s / torch.sum(s)
    p = p[p > 1e-15]
    
    if len(p) == 0:
        return 0.0, 0.0
    
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(entropy)
    
    return effective_rank.item(), entropy.item()


def run_full_block_test(config: Config) -> Dict:
    """运行完整残差块测试"""
    
    prompts = {
        "High Logic (Math)": "Solve this step by step: If 3x + 5 = 20, what is the value of x^2?",
        "High Noise (Random)": "asdfjkl qwer uiop zxcv bnm ghjkl tyu rty fgh vbn",
        "High Redundancy (Repeat)": "The the the the the the the the the the the the",
        "Normal Dialogue": "Hello, how are you doing today? I would like to know about"
    }
    
    model, tokenizer, device, dtype = load_model_from_comfyui(config)
    
    num_layers = model.config.num_hidden_layers
    
    test_layers = [0, 2, num_layers - 2]
    test_layers = sorted(list(set(test_layers)))
    test_layers = [l for l in test_layers if l < num_layers - 1]
    
    results = {
        "layers": test_layers,
        "lambda_max": {name: [] for name in prompts.keys()},
        "er": {name: [] for name in prompts.keys()},
        "entropy": {name: [] for name in prompts.keys()},
    }
    
    for prompt_name, text in prompts.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {prompt_name}")
        logger.info(f"{'='*60}")
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        for layer_idx in test_layers:
            logger.info(f"\n  Layer {layer_idx}")
            
            try:
                lambda_max, rayleigh = power_iteration_max_eigenvalue(
                    model, inputs, layer_idx, device, dtype, config.n_power_iter
                )
                
                S, max_sv = estimate_jacobian_spectrum_samples(
                    model, inputs, layer_idx, device, dtype, config.n_samples
                )
                
                if S is not None and len(S) > 0:
                    er, entropy = calculate_effective_rank(S)
                else:
                    er, entropy = 0.0, 0.0
                
                results["lambda_max"][prompt_name].append(lambda_max)
                results["er"][prompt_name].append(er)
                results["entropy"][prompt_name].append(entropy)
                
                status = "扩张" if lambda_max > 1.1 else ("收缩" if lambda_max < 0.9 else "临界")
                logger.info(f"    λ_max = {lambda_max:.4f} ({status})")
                logger.info(f"    ER = {er:.2f}, Entropy = {entropy:.4f}")
                
            except Exception as e:
                logger.error(f"    Layer {layer_idx} 失败: {e}")
                results["lambda_max"][prompt_name].append(0.0)
                results["er"][prompt_name].append(0.0)
                results["entropy"][prompt_name].append(0.0)
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    return results, model.config.num_hidden_layers


def visualize_results(results: Dict, num_layers: int, model_name: str, save_path: Optional[Path] = None):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#2ecc71']
    markers = ['o', 's', '^', 'D']
    
    ax1, ax2, ax3 = axes
    
    for i, (prompt_name, lambda_list) in enumerate(results["lambda_max"].items()):
        if lambda_list:
            ax1.plot(results["layers"][:len(lambda_list)], lambda_list,
                    marker=markers[i], linewidth=2.5, color=colors[i],
                    label=prompt_name, alpha=0.8, markersize=8)
    
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.fill_between([0, num_layers], [1.1, 1.1], [2, 2], alpha=0.1, color='red', label='扩张区')
    ax1.fill_between([0, num_layers], [0, 0], [0.9, 0.9], alpha=0.1, color='green', label='收缩区')
    ax1.set_title('Max Eigenvalue $\\lambda_{max}$ of Full Block Jacobian', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('$\\lambda_{max}$', fontsize=12)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, num_layers - 0.5)
    
    for i, (prompt_name, er_list) in enumerate(results["er"].items()):
        if er_list:
            ax2.plot(results["layers"][:len(er_list)], er_list,
                    marker=markers[i], linewidth=2.5, color=colors[i],
                    label=prompt_name, alpha=0.8, markersize=8)
    
    ax2.set_title('Effective Rank of Jacobian Spectrum', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Effective Rank', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    for i, (prompt_name, entropy_list) in enumerate(results["entropy"].items()):
        if entropy_list:
            ax3.plot(results["layers"][:len(entropy_list)], entropy_list,
                    marker=markers[i], linewidth=2.5, color=colors[i],
                    label=prompt_name, alpha=0.8, markersize=8)
    
    ax3.set_title('Shannon Entropy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Layer Index', fontsize=12)
    ax3.set_ylabel('Entropy (nat)', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f'Full Residual Block Jacobian Analysis - {model_name}', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图表已保存至: {save_path}")
    
    plt.show()
    plt.close(fig)


def print_summary(results: Dict, num_layers: int):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("完整残差块 Jacobian 谱测试结果")
    print("="*70)
    
    print("\n各提示类型的谱特性:")
    print("-"*70)
    print(f"{'Prompt Type':<25} {'Mean λ':>10} {'Mean ER':>10} {'结论':>15}")
    print("-"*70)
    
    for prompt_name in results["lambda_max"].keys():
        lambda_list = [x for x in results["lambda_max"][prompt_name] if x > 0]
        er_list = [x for x in results["er"][prompt_name] if x > 0]
        
        if lambda_list:
            mean_lambda = np.mean(lambda_list)
            mean_er = np.mean(er_list) if er_list else 0
            
            if mean_lambda > 1.1:
                conclusion = "**扩张系统**"
            elif mean_lambda < 0.9:
                conclusion = "收缩系统"
            else:
                conclusion = "临界系统"
            
            print(f"{prompt_name:<25} {mean_lambda:>10.4f} {mean_er:>10.2f} {conclusion:>15}")
    
    print("\n层间详细数据:")
    print("-"*70)
    
    for prompt_name in results["lambda_max"].keys():
        print(f"\n{prompt_name}:")
        for i, layer in enumerate(results["layers"]):
            if i < len(results["lambda_max"][prompt_name]):
                lam = results["lambda_max"][prompt_name][i]
                er = results["er"][prompt_name][i]
                if lam > 0:
                    status = "↑" if lam > 1.1 else ("↓" if lam < 0.9 else "≈")
                    print(f"  Layer {layer:2d}: λ = {lam:.4f} {status}, ER = {er:.2f}")
    
    print("\n" + "="*70)
    print("Layer 2 专项分析 (High Redundancy):")
    print("-"*70)
    
    layer_2_idx = None
    for i, l in enumerate(results["layers"]):
        if l == 2:
            layer_2_idx = i
            break
    
    if layer_2_idx is not None:
        for prompt_name in results["lambda_max"].keys():
            if layer_2_idx < len(results["lambda_max"][prompt_name]):
                lam = results["lambda_max"][prompt_name][layer_2_idx]
                er = results["er"][prompt_name][layer_2_idx]
                if lam > 0:
                    print(f"  {prompt_name}: λ = {lam:.4f}, ER = {er:.2f}")
                    if prompt_name == "High Redundancy (Repeat)":
                        if lam < 1:
                            print(f"    → Layer 2 在冗余输入下是**收缩的**，主动熄灭信号！")
                        else:
                            print(f"    → Layer 2 在冗余输入下仍然扩张")


def main():
    print("="*70)
    print("完整残差块 Jacobian 谱测试 - 决定性实验 (v2)")
    print("="*70)
    
    config = Config(
        model_name="Qwen2.5-0.5B",
        n_power_iter=30,
        n_samples=50,
        use_float64=True
    )
    
    print(f"\n配置:")
    print(f"  模型: {config.model_name}")
    print(f"  Power Iteration 迭代数: {config.n_power_iter}")
    print(f"  随机采样数: {config.n_samples}")
    print(f"  精度: float64")
    
    print(f"\n判断标准:")
    print(f"  λ > 1.1  → 扩张系统 (CCF 强形式被否定)")
    print(f"  λ ≈ 1.0  → 临界系统")
    print(f"  λ < 0.9  → 收缩系统 (CCF 强形式成立)")
    
    results, num_layers = run_full_block_test(config)
    
    print_summary(results, num_layers)
    
    save_path = Path(__file__).parent.parent / "validation_results" / f"full_block_jacobian_v2_{config.model_name.replace('.', '-')}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_results(results, num_layers, config.model_name, save_path)
    
    print("\n" + "="*70)
    print("实验完成")
    print("="*70)
    
    all_lambda = []
    for lambda_list in results["lambda_max"].values():
        all_lambda.extend([x for x in lambda_list if x > 0])
    
    if all_lambda:
        mean_lambda = np.mean(all_lambda)
        print(f"\n最终结论:")
        print(f"  平均 λ_max = {mean_lambda:.4f}")
        
        if mean_lambda > 1.1:
            print(f"  → Transformer 层是**扩张系统**")
            print(f"  → CCF 强形式（正交方向收缩）被否定")
            print(f"  → 支持 Controlled Critical Expansion (CCE) 理论")
        elif mean_lambda < 0.9:
            print(f"  → Transformer 层是**收缩系统**")
            print(f"  → CCF 强形式成立")
        else:
            print(f"  → Transformer 层是**临界系统**")


if __name__ == "__main__":
    main()
