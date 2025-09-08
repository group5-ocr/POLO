# -*- coding: utf-8 -*-
# 범용 선 그래프(학습곡선, ROC/PR 등). seaborn 미사용, 색상 미지정.
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_curve_generic(inputs, out_path):
    series = inputs.get("series", [])
    plt.figure(figsize=(5.2, 4.0))  # 각 차트는 개별 Figure
    for s in series:
        x = s.get("x", [])
        y = s.get("y", [])
        label = s.get("label", None)
        plt.plot(x, y, marker="o", label=label)
    if any(s.get("label") for s in series):
        plt.legend()
    plt.xlabel(inputs.get("xlabel", ""))
    plt.ylabel(inputs.get("ylabel", ""))
    plt.title(inputs.get("title", ""))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("curve_generic", ["series"], ["xlabel","ylabel","title"], render_curve_generic))
