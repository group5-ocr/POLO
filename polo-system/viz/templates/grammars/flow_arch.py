# -*- coding: utf-8 -*-
# 흐름/아키텍처 다이어그램 문법. pos가 없으면 자동 일렬 배치.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from registry import Grammar, register

def _draw_node(ax, x, y, w, h, text):
    # 중앙 (x,y)에 w*h 사각형과 텍스트를 그립니다.
    rect = Rectangle((x - w/2, y - h/2), w, h, fill=False)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=9)

def _draw_arrow(ax, x1, y1, x2, y2):
    # (x1,y1) -> (x2,y2) 방향 화살표
    arr = FancyArrowPatch((x1,y1), (x2,y2), arrowstyle="->", mutation_scale=10, linewidth=1.5)
    ax.add_patch(arr)

def render_flow_arch(inputs, out_path):
    nodes = inputs.get("nodes", [])
    edges = inputs.get("edges", [])
    if not any("pos" in n for n in nodes):
        for i, n in enumerate(nodes):
            n["pos"] = [ (i+1)/(len(nodes)+1), 0.5 ]

    plt.figure(figsize=(6.2, 3.8))
    ax = plt.gca()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks([]); ax.set_yticks([])

    centers = {}
    for n in nodes:
        x, y = n["pos"]
        centers[n["id"]] = (x,y)
        _draw_node(ax, x, y, 0.18, 0.12, n.get("label", n["id"]))

    for e in edges:
        if e["src"] in centers and e["dst"] in centers:
            x1,y1 = centers[e["src"]]
            x2,y2 = centers[e["dst"]]
            _draw_arrow(ax, x1+0.09, y1, x2-0.09, y2)

    plt.title(inputs.get("title","Flow / Architecture"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("flow_arch", ["nodes","edges"], ["title"], render_flow_arch))
