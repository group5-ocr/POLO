import numpy as np
import matplotlib.pyplot as plt
from registry import Grammar, register

def _sigmoid(z):  # overflow 안전
    z = np.clip(z, -60, 60)
    return 1.0/(1.0+np.exp(-z))

def _softplus(z):
    z = np.clip(z, -60, 60)
    return np.log1p(np.exp(z))

def _gelu(z):
    # Hendrycks & Gimpel (tanh 근사)
    return 0.5*z*(1.0+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*(z**3))))

def _selu(z):
    # 공식 상수
    alpha, scale = 1.6732632423543772, 1.0507009873554805
    return scale*np.where(z>0, z, alpha*(np.exp(z)-1))

def _relu6(z):
    return np.clip(z, 0, 6)

def _hard_sigmoid(z):
    return np.clip(0.2*z+0.5, 0.0, 1.0)

def _hard_swish(z):
    return z*_hard_sigmoid(z)

def _hard_tanh(z):
    return np.clip(z, -1.0, 1.0)

def _apply_activation(name: str, x: np.ndarray, params: dict) -> np.ndarray:
    n = name.lower()
    if n == "relu":       return np.maximum(0, x)
    if n == "leakyrelu":  return np.where(x>=0, x, (params.get("alpha",0.2))*x)
    if n == "prelu":      return np.where(x>=0, x, (params.get("alpha",0.25))*x)
    if n == "tanh":       return np.tanh(x)
    if n == "sigmoid":    return _sigmoid(x)
    if n == "elu":        return np.where(x>=0, x, (params.get("alpha",1.0))*(np.exp(x)-1))
    if n == "selu":       return _selu(x)
    if n == "gelu":       return _gelu(x)
    if n == "silu":       return x*_sigmoid((params.get("beta",1.0))*x)  # Swish/SiLU
    if n == "softplus":   return _softplus(x)
    if n == "mish":       return x*np.tanh(_softplus(x))
    if n == "hardsigmoid":return _hard_sigmoid(x)
    if n == "hardswish":  return _hard_swish(x)
    if n == "hardtanh":   return _hard_tanh(x)
    if n == "relu6":      return _relu6(x)
    # fallback
    return np.maximum(0, x)

def render_activations_panel(inputs, out_path):
    funcs = inputs.get("funcs", [])
    if not funcs:  # 아무 것도 없으면 그리지 않음
        return
    xlim = inputs.get("xlim", [-6, 6]); ylim = inputs.get("ylim", [-1.3, 1.3])
    title = inputs.get("title", "Activation functions")

    xs = np.linspace(xlim[0], xlim[1], 801)

    plt.figure(figsize=(6.8, 4.4), dpi=200)
    for f in funcs:
        name = f.get("name","ReLU")
        params = {k:v for k,v in f.items() if k != "name"}
        ys = _apply_activation(name, xs, params)

        # 범례에 파라미터 표시
        label = name
        if "alpha" in params:
            label += f"(α={params['alpha']:.2g})"
        if "beta" in params:
            label += f"(β={params['beta']:.2g})"
        plt.plot(xs, ys, label=label)

    plt.axhline(0, color="#999", lw=0.6); plt.axvline(0, color="#999", lw=0.6)
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

register(Grammar("activations_panel", ["funcs"], ["title","xlim","ylim"], render_activations_panel))