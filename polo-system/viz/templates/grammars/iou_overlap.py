import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register
import math

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / float(aw*ah + bw*bh - inter + 1e-9)

def _place_B_for_iou(A, target_iou: float):
    ax, ay, aw, ah = A
    a = aw * ah
    if target_iou >= 0.999:  # 완전 겹침
        return [ax, ay, aw, ah]
    # O = IoU*(A+B)/(1+IoU) ; A=B=a
    inter = (2 * a * target_iou) / (1.0 + target_iou)
    inter = max(1e-6, min(inter, a))
    side = math.sqrt(inter)             # 정사각형 겹침 가정
    dx = max(0.0, aw - side)
    dy = max(0.0, ah - side)
    bx = min(1 - aw, ax + dx)
    by = min(1 - ah, ay + dy)
    return [bx, by, aw, ah]

def render_iou_overlap(inputs, out_path):
    A = inputs.get("A", [0.15, 0.15, 0.55, 0.55])
    B = inputs.get("B", [0.35, 0.25, 0.55, 0.55])
    title = inputs.get("title", "IoU overlap")
    iou_val = inputs.get("iou", None)              # 0~1 (있으면 사용)
    show_value = bool(inputs.get("show_value", iou_val is not None))
    show_example = bool(inputs.get("example_badge", True))

    plt.figure(figsize=(4.8, 5.0))
    ax = plt.gca()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])

    ax.add_patch(Rectangle((A[0], A[1]), A[2], A[3], color="#f5c16c", alpha=0.5))
    ax.add_patch(Rectangle((B[0], B[1]), B[2], B[3], color="#f0a63a", alpha=0.5))

    if show_value and isinstance(iou_val, (int, float)):
        ax.text(0.04, 0.95, f"IoU ≈ {float(iou_val):.2f}",
                fontsize=12, ha="left", va="top")
    elif not show_value and show_example:
        ax.text(0.96, 0.04, "예시", fontsize=10, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999"))

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar(
    "iou_overlap",
    [],  # required 없음
    ["A", "B", "title", "show_value", "iou", "example_badge"],
    render_iou_overlap
))