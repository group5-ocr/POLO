import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from registry import Grammar, register
import textwrap

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

# 설명이 의미 있을 때만 간결 캡션 (정의/비교 중심)
def _needs_explain(names: set) -> bool:
    return len(names) >= 2  # 여러 개일 때만 비교 설명

def _explain(names: set, funcs_spec: list, xlim):
    n = {s.lower() for s in names}
    lines = []
    if "relu" in n:  lines.append("ReLU: x<0→0, x>0 선형 — 비포화/희소, dead ReLU 가능.")
    if ("leakyrelu" in n) or ("prelu" in n):
        lines.append("Leaky/PReLU: 음수 기울기 유지(α)로 dead ReLU 완화.")
    if "softplus" in n: lines.append("Softplus: ReLU의 매끄러운 근사(log(1+e^x)).")
    if ("silu" in n) or ("mish" in n) or ("gelu" in n):
        lines.append("SiLU/Mish/GELU: 부드러운 ReLU 계열 — 작은 음수도 완만 통과.")
    if ("elu" in n) or ("selu" in n):
        lines.append("ELU/SELU: 음수 포화(바이어스 보정), SELU는 self-normalizing.")
    if "tanh" in n:  lines.append("tanh: [-1,1] 포화, 0-중심 — sigmoid보다 학습 안정적.")
    if ("sigmoid" in n) and ("tanh" not in n):
        lines.append("Sigmoid: [0,1] 포화, 0-비중심 — 기울기 소실 쉬움.")
    if "relu" in n and (("silu" in n) or ("gelu" in n) or ("softplus" in n)):
        lines.append("비교: ReLU는 음수 완전 차단, SiLU/GELU/Softplus는 완만 통과 → 더 부드러운 학습.")
    if "relu" in n and (("leakyrelu" in n) or ("prelu" in n)):
        lines.append("비교: Leaky/PReLU는 음수 기울기 유지 → dead ReLU 리스크 감소.")
    if "tanh" in n and "sigmoid" in n:
        lines.append("비교: tanh(0-중심) vs sigmoid(0-비중심) — tanh가 수렴 유리한 경우 多.")
    if "elu" in n and "selu" in n:
        lines.append("비교: SELU는 고정 α/scale로 층 분포 안정화 목표.")

    # α/β 파라미터 노트
    alphas = []
    for f in funcs_spec:
        nm=f.get("name","").lower()
        if nm in ("leakyrelu","prelu") and "alpha" in f: alphas.append(f"α={f['alpha']}")
        if nm=="silu" and "beta" in f: alphas.append(f"β={f['beta']}")
        if nm=="elu" and "alpha" in f: alphas.append(f"ELU α={f['alpha']}")
    if alphas:
        lines.append("파라미터: " + ", ".join(alphas))

    if not lines:
        return None

    # 항목별 wrap + 하드 개행 유지, 최대 3줄까지만
    max_lines = 3
    wrapped = []
    for ln in lines:
        for seg in textwrap.wrap(ln, width=76, break_long_words=False):
            wrapped.append(seg)
            if len(wrapped) >= max_lines:
                return "\n".join(wrapped) + "\n"   # 끝에 줄바꿈 1개 패딩
    return "\n".join(wrapped) + "\n"

def render_activations_panel(inputs, out_path):
    funcs = inputs.get("funcs", [])
    if not funcs:  # 아무 것도 없으면 그리지 않음
        return
    xlim = inputs.get("xlim", [-6, 6]); ylim = inputs.get("ylim", [-1.3, 1.3])
    title = inputs.get("title", "Activation functions")

    xs = np.linspace(xlim[0], xlim[1], 801)

    # 캡션 유무에 따라 1단/2단 레이아웃 선택
    names = {f.get("name","") for f in funcs}
    manual_caption = (inputs.get("caption") or "").strip() if isinstance(inputs.get("caption"), str) else None
    auto_caption = None
    if not manual_caption and _needs_explain({s.lower() for s in names}):
        auto_caption = _explain(names, funcs, xlim)
    caption = manual_caption or auto_caption

    if caption:
        fig = plt.figure(figsize=(6.8, 5.2), dpi=200)  # 캡션 줄 추가
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 0.28], hspace=0.15)
        ax = fig.add_subplot(gs[0])
        cap_ax = fig.add_subplot(gs[1]); cap_ax.axis("off")
    else:
        fig = plt.figure(figsize=(6.8, 4.4), dpi=200)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

    # 본 그림
    present_names = set()
    for f in funcs:
        name = f.get("name","ReLU"); present_names.add(name)
        params = {k:v for k,v in f.items() if k != "name"}
        ys = _apply_activation(name, xs, params)
        label = name
        if "alpha" in params: label += f"(α={params['alpha']:.2g})"
        if "beta"  in params: label += f"(β={params['beta']:.2g})"
        ax.plot(xs, ys, label=label)

    ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    leg = ax.legend(fontsize=9)
    if leg: leg.set_draggable(True)
    ax.set_title(title)

    # 캡션 축 (플롯 밖, 겹침 없음)
    if caption:
        cap_ax.text(0.5, 0.95, caption + "\n",  # ← 끝에 \n 한 줄 패딩
            ha="center", va="top", fontsize=9, wrap=True)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

register(Grammar("activations_panel",
    ["funcs"],
    ["title","xlim","ylim","caption"],
    render_activations_panel))