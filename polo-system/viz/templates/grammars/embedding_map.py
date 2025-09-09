# -*- coding: utf-8 -*-
# 2D 임베딩 맵 문법 (점 + 선택적 라벨)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_embedding_map(inputs, out_path):
    points = inputs.get("points", [])
    plt.figure(figsize=(5.2,5.0))
    for p in points:
        plt.scatter([p["x"]],[p["y"]], s=12, alpha=0.8)
        if "label" in p:
            plt.text(p["x"]+0.01, p["y"]+0.01, str(p["label"]), fontsize=8)
    plt.xlabel(inputs.get("xlabel","dim1"))
    plt.ylabel(inputs.get("ylabel","dim2"))
    plt.title(inputs.get("title","Embedding Map"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("embedding_map", ["points"], ["xlabel","ylabel","title"], render_embedding_map))
