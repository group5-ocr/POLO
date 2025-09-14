# 수직 스택 블록(모듈 구성) 문법
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register

def render_module_block(inputs, out_path):
    blocks = inputs.get("blocks", [])
    gap = 0.06  # 블록 간 간격
    h = 0.12    # 각 블록 높이
    total_h = len(blocks)*h + (len(blocks)-1)*gap
    y0 = 0.5 + total_h/2 - h/2

    plt.figure(figsize=(4.2, 5.2))
    ax = plt.gca()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks([]); ax.set_yticks([])

    for i, name in enumerate(blocks):
        y = y0 - i*(h+gap)
        rect = Rectangle((0.25, y - h/2), 0.5, h, fill=False)
        ax.add_patch(rect)
        ax.text(0.5, y, name, ha="center", va="center")
        if i < len(blocks)-1:
            # 아래 방향 화살표
            ax.annotate("", xy=(0.5, y - h/2), xytext=(0.5, y - h/2 - gap + 0.01),
                        arrowprops=dict(arrowstyle="->", linewidth=1.5))

    plt.title(inputs.get("title","Module Blocks"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("module_block", ["blocks"], ["title"], render_module_block))
