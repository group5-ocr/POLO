# BER vs SNR 비교 그래프 (Deepcode vs Modulo-SK)
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
from matplotlib import rcParams
from registry import register, Grammar

rcParams['axes.unicode_minus'] = False

def _ensure_inputs(spec: Dict[str, Any]) -> Dict[str, Any]:
    return spec.get("inputs", spec)

def render_ber_vs_snr(spec: Dict[str, Any], save_path: str) -> str:
    spec = _ensure_inputs(spec)
    series: List[Dict[str, Any]] = spec["series"]

    title   = spec.get("title", "BER vs SNR")
    x_label = spec.get("x_label", "SNR (dB)")
    y_label = spec.get("y_label", "BER")
    y_log   = bool(spec.get("y_log", True))

    fig = plt.figure(figsize=(7.5, 5.2), dpi=160)
    ax = fig.add_subplot(111)
    ax.set_title(title, fontweight="bold", fontsize=16, pad=10)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    for s in series:
        pts = s.get("points") or []
        if not pts:
            continue
        name = s.get("name", "series")
        xs = [p[0] for p in pts]
        ys = [max(1e-12, p[1]) for p in pts]
        if len(pts) >= 2:
            ax.plot(xs, ys, marker="o", label=name)
        else:
            ax.scatter(xs[0], ys[0], label=name)

    ax.legend(frameon=True, loc="best")
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path

register(Grammar(
    "ber_vs_snr",
    ["inputs"],
    ["x_label", "y_label", "title", "y_log", "notes", "caption", "caption_bottom", "caption_y"],
    render_ber_vs_snr
))