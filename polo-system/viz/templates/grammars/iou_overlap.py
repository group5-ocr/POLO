import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / float(aw*ah + bw*bh - inter + 1e-9)

def render_iou_overlap(inputs, out_path):
    A = inputs.get("A", [0.15,0.15,0.55,0.55])  # [x,y,w,h], 0~1
    B = inputs.get("B", [0.35,0.25,0.55,0.55])
    title = inputs.get("title", "IoU overlap")
    show_val = inputs.get("show_value", True)

    plt.figure(figsize=(4.8,5.0))
    ax = plt.gca()
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks([]); ax.set_yticks([])

    ax.add_patch(Rectangle((A[0],A[1]),A[2],A[3],color="#f5c16c",alpha=0.5))
    ax.add_patch(Rectangle((B[0],B[1]),B[2],B[3],color="#f0a63a",alpha=0.5))
    if show_val:
        ax.text(0.04,0.95,f"IoU ≈ {_iou(A,B):.2f}",fontsize=12,ha="left",va="top")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("iou_overlap", [], ["A","B","title","show_value"], render_iou_overlap))