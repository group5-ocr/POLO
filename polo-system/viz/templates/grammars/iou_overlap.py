import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register
import math

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter  = iw * ih
    union  = aw*ah + bw*bh - inter
    if union <= 0:
        return 0.0
    v = inter / union
    return max(0.0, min(1.0, v))  # 0~1 클램프

def render_iou_overlap(inputs, out_path):
    A = inputs.get("A", [0.15, 0.15, 0.55, 0.55])
    B = inputs.get("B", [0.35, 0.25, 0.55, 0.55])

    # 받은 iou가 신뢰 구간(0.01~0.99)이면 사용, 아니면 박스 A/B로 계산
    iou_from_input = inputs.get("iou", None)
    if isinstance(iou_from_input, (int, float)) and 0.01 < float(iou_from_input) < 0.99:
        iou_to_show = float(iou_from_input)
    else:
        iou_to_show = _iou(A, B)

    title       = inputs.get("title", "IoU(겹침) 예시")
    show_value  = bool(inputs.get("show_value", True))
    show_badge  = bool(inputs.get("example_badge", iou_from_input is None))

    plt.figure(figsize=(4.8, 5.0))
    ax = plt.gca(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks([]); ax.set_yticks([])

    ax.add_patch(Rectangle((A[0], A[1]), A[2], A[3],
                        facecolor="#f5c16c", alpha=0.50,
                        edgecolor="#b07d2a", linewidth=1.5, zorder=1))
    ax.add_patch(Rectangle((B[0], B[1]), B[2], B[3],
                        facecolor="#f0a63a", alpha=0.50,
                        edgecolor="#945c13", linewidth=1.5, zorder=2))

    if show_value:
        ax.text(0.04, 0.95, f"IoU ≈ {iou_to_show:.2f}",
                fontsize=12, ha="left", va="top")
    elif show_badge:
        ax.text(0.96, 0.04, "예시",
                fontsize=10, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999"))

    plt.title(title)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig = plt.gcf()
    plt.figtext(
        0.5, 0.005,
        "IoU = |A∩B| / |A∪B|  ·  겹치는 면적이 없으면 0, 완전 일치면 1",
        ha="center", va="bottom", fontsize=9
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.10)
    plt.close()

register(Grammar(
    "iou_overlap",
    [],  # required 없음
    ["A", "B", "title", "show_value", "iou", "example_badge"],
    render_iou_overlap
))