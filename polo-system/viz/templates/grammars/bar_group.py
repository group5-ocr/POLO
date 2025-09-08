# -*- coding: utf-8 -*-
# 그룹 막대 그래프 문법
import numpy as np
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_bar_group(inputs, out_path):
    categories = inputs.get("categories", [])
    series = inputs.get("series", [])
    n = len(categories)
    m = len(series)
    # 비어 있으면 간단 폴백 사용
    if n == 0 or m == 0:
        categories = ["A","B","C"]
        series = [{"label":"S1","values":[1,2,3]}]
        n, m = 3, 1

    width = 0.8 / max(1, m)
    idx = np.arange(n)
    plt.figure(figsize=(5.6, 4.0))
    for i, s in enumerate(series):
        values = s.get("values", [0]*n)
        x = idx + (i - (m-1)/2)*width
        plt.bar(x, values, width=width, label=s.get("label"))
    if any(s.get("label") for s in series):
        plt.legend()
    plt.xticks(idx, categories, rotation=0)
    plt.title(inputs.get("title", ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("bar_group", ["categories","series"], ["title"], render_bar_group))
