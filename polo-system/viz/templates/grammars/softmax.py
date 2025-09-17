import numpy as np
import matplotlib.pyplot as plt
from registry import Grammar, register

def _softmax(logits, tau=1.0):
    z = np.array(logits, dtype=float) / float(tau)
    z -= z.max()                    
    ez = np.exp(z)
    return ez / ez.sum()

def render_softmax(inputs, out_path):
    # inputs
    taus = inputs.get("taus", [1.0])                  # e.g., [1.0] or [0.5,1.0,2.0]
    title = inputs.get("title", {"en":"Softmax","ko":"소프트맥스"})
    xr = inputs.get("logit_range", [-6, 6])           
    x = np.linspace(float(xr[0]), float(xr[1]), 400)

    plt.figure(figsize=(6.8, 4.0))
    for tau in taus:
        # 2-class case: prob of class0 when logits=[x, 0]  (softmax==sigmoid for 2-class)
        y = np.array([_softmax([v, 0.0], tau=tau)[0] for v in x])
        plt.plot(x, y, label=(f"τ={tau:g}"))

    plt.axhline(0.5, lw=0.6, color="#888", alpha=0.6)
    plt.axvline(0.0, lw=0.6, color="#888", alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("logit Δ (z₁ − z₂)")
    plt.ylabel("P(class 1)")
    plt.title(title.get("ko") if isinstance(title, dict) else title)
    if len(taus) > 1:
        plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    cap = None
    try:
        # title은 dict 또는 str일 수 있음. (그대로 유지)
        if isinstance(title, dict):
            _t = (title.get("ko") or title.get("en") or "Softmax")
        else:
            _t = str(title)

        if isinstance(taus, (list, tuple)) and len(taus) > 1:
            cap = "T↑ → 확률분포 평탄화,  T↓ → 날카로움 증가  (x축: 클래스 간 로그확률 차)"
        else:
            cap = "소프트맥스: z/τ → 확률 (안정화를 위해 최대 로짓을 빼고 계산)"
    except Exception:
        cap = None

    if cap:
        plt.figtext(0.5, 0.008, cap, ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close()

# 포지셔널 인자 4개로 등록
register(Grammar(
    "softmax",
    ["taus"],                               # required fields
    ["title", "logit_range", "class_labels"],  # optional fields
    render_softmax
))