import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register
import numpy as np

def _tuplify_wh(v, default=(640, 640)):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    if isinstance(v, (int, float)):
        return float(v), float(v)
    return float(default[0]), float(default[1])

def _pair_grid(g):
    # g: 13 또는 [13,13] 또는 (26,26)
    if isinstance(g, (list, tuple)) and len(g) == 2:
        return int(g[0]), int(g[1])
    return int(g), int(g)

def render_cell_scale(inputs, out_path):
    """
    두 해상도(그리드)에서 같은 '셀 내부(norm) 예측'이
    어떻게 '픽셀 박스' 크기로 변하는지 시각화.
    """
    img_w, img_h = _tuplify_wh(inputs.get("img_wh", [640, 640]))
    grids = inputs.get("grids", [[13, 13], [26, 26]])  # 두 개 권장
    grids = [ _pair_grid(g) for g in grids[:2] ] or [(13,13), (26,26)]

    # 어느 셀에서 예측하는지 (기본: 중심 셀)
    which_cell = inputs.get("cell_xy", None)  # [cx, cy] in integer cell index
    norm_center = inputs.get("norm_center", [0.5, 0.5])   # [0..1] in cell
    norm_wh_cell = inputs.get("norm_wh_cell", [0.7, 0.7]) # box (w,h) as cell fraction

    # 앵커 반영(옵션): 셀 폭/높이에 대한 배수
    # 예: [1.2, 0.8] 이면, box_w = norm_wh_cell*w_cell*1.2
    anchor_rel_cell = inputs.get("anchor_rel_cell", None)

    example_badge = bool(inputs.get("example_badge", False))
    title = inputs.get("title", "Cell-relative → Pixel-scale")

    # 패널 개수는 grids 길이에 맞춤(최소 1, 최대 2)
    ncols = max(1, min(2, len(grids) or 2))
    fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 4.4), dpi=200)
    # ndarray 또는 단일 Axis 모두 1-D 리스트로 변환
    if isinstance(axs, np.ndarray):
        axs = axs.ravel().tolist()
    else:
        axs = [axs]

    for ax, (gw, gh) in zip(axs, (grids[:ncols] if grids else [(13,13), (26,26)][:ncols])):
        ax.set_xlim(0, img_w); ax.set_ylim(0, img_h)
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

        # 셀 크기
        cell_w, cell_h = img_w / gw, img_h / gh

        # 시각적으로 중앙 셀을 기본으로
        if which_cell and len(which_cell) == 2:
            cx_i, cy_i = int(which_cell[0]), int(which_cell[1])
        else:
            cx_i, cy_i = gw // 2, gh // 2

        # 셀 좌상단 픽셀 좌표
        cell_x0, cell_y0 = cx_i * cell_w, cy_i * cell_h

        # norm → 픽셀 중심
        cx_norm, cy_norm = float(norm_center[0]), float(norm_center[1])
        box_cx = cell_x0 + cx_norm * cell_w
        box_cy = cell_y0 + cy_norm * cell_h

        # 셀 대비 박스 크기
        bw = float(norm_wh_cell[0]) * cell_w
        bh = float(norm_wh_cell[1]) * cell_h
        if anchor_rel_cell and len(anchor_rel_cell) == 2:
            bw *= float(anchor_rel_cell[0])
            bh *= float(anchor_rel_cell[1])

        # 박스 그리기 (중심 → 좌상단 변환)
        rect = Rectangle((box_cx - bw/2, box_cy - bh/2), bw, bh,
                         facecolor="#f0a63a", alpha=0.45, edgecolor="#945c13", lw=1.5)
        ax.add_patch(rect)

        # 그리드 라인(간략)
        for i in range(gw+1):
            ax.axvline(i*cell_w, color="#cccccc", lw=0.6, zorder=0)
        for j in range(gh+1):
            ax.axhline(j*cell_h, color="#cccccc", lw=0.6, zorder=0)

        # 강조 셀
        ax.add_patch(Rectangle((cell_x0, cell_y0), cell_w, cell_h,
                               fill=False, edgecolor="#333", lw=1.2))

        # 주석
        ax.text(6, 18, f"{gw}×{gh} grid", fontsize=10, va="top")
        ax.text(6, 36, f"cell={cx_i},{cy_i}  size≈({bw:.1f}px, {bh:.1f}px)",
                fontsize=9, va="top", color="#333")

        if example_badge:
            ax.text(img_w-6, img_h-6, "예시", ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999"))

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

register(Grammar(
    "cell_scale",
    [],  # required
    ["img_wh", "grids", "cell_xy", "norm_center", "norm_wh_cell",
     "anchor_rel_cell", "example_badge", "title"],
    render_cell_scale
))