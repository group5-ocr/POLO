# 임베딩 합성 도식 (Token + Segment + Position → Encoder)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register

def render_embedding_sum(inputs, out_path):
    rows = inputs.get("rows", ["Token","Segment(A/B)","Position"])
    right = inputs.get("right", "Encoder")
    title = inputs.get("title", "Embeddings → Encoder")

    plt.figure(figsize=(5.6, 3.0))
    ax = plt.gca(); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    y0 = 0.72; h = 0.18; gap = 0.06
    # 왼쪽 3개 박스
    for i, name in enumerate(rows):
        y = y0 - i*(h+gap)
        ax.add_patch(Rectangle((0.08, y-h/2), 0.32, h, fill=False))
        ax.text(0.24, y, name, ha="center", va="center")

    # Sum 박스
    ax.add_patch(Rectangle((0.46, 0.5-h/2), 0.12, h, fill=False))
    ax.text(0.52, 0.5, "Sum", ha="center", va="center")

    # + 표식
    ax.text(0.42, 0.72, "+", ha="center", va="center", fontsize=14)
    ax.text(0.42, 0.48, "+", ha="center", va="center", fontsize=14)

    # 오른쪽 Encoder
    ax.add_patch(Rectangle((0.68, 0.5-h/2), 0.26, h, fill=False))
    ax.text(0.81, 0.5, right, ha="center", va="center")
    ax.annotate("", xy=(0.68,0.5), xytext=(0.58,0.5),
                arrowprops=dict(arrowstyle="->", linewidth=1.8))

    plt.title(title if isinstance(title, str) else (title.get("ko") or title.get("en") or ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("embedding_sum", ["rows"], ["title","right"], render_embedding_sum))