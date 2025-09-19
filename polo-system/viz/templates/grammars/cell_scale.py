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
    if isinstance(g, (list, tuple)) and len(g) == 2:
        return int(g[0]), int(g[1])
    return int(g), int(g)

def render_cell_scale(inputs, out_path):
    """
    두 해상도(그리드)에서 같은 '셀 내부(norm) 예측'이
    어떻게 '픽셀 박스' 크기로 변하는지 시각화.
    (트리거/정규식 전혀 없음. inputs만 사용)
    """
    img_w, img_h = _tuplify_wh(inputs.get("img_wh", [640, 640]))
    grids = inputs.get("grids", [[13, 13], [26, 26]])
    grids = [_pair_grid(g) for g in grids[:2]] or [(13, 13), (26, 26)]

    which_cell      = inputs.get("cell_xy", None)            # (i,j) 셀 인덱스
    norm_center     = inputs.get("norm_center", [0.5, 0.5])  # 셀 내부 중심 (0~1)
    norm_wh_cell    = inputs.get("norm_wh_cell", [0.7, 0.7]) # 셀 상대 크기 (0~1)
    anchor_rel_cell = inputs.get("anchor_rel_cell", None)    # (aw, ah) 선택

    # 시각 옵션
    zoom       = float(inputs.get("zoom", 1.35))
    dpi        = int(inputs.get("dpi", 240))
    grid_lw    = float(inputs.get("grid_lw", 0.9))

    # figure 외곽 캡션 위치
    caption_top_pad    = float(inputs.get("caption_top_pad", 0.010))
    caption_bottom_pad = float(inputs.get("caption_bottom_pad", 0.015))
    title_y            = float(inputs.get("title_y", 0.988))
    title              = inputs.get("title", "셀 그리드 → 픽셀 스케일")

    ncols = max(1, min(2, len(grids)))
    per_w, per_h = 6.0 * zoom, 5.6 * zoom
    fig, axs = plt.subplots(1, ncols, figsize=(per_w * ncols, per_h), dpi=dpi)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.10, wspace=0.12)

    axs = axs.ravel().tolist() if isinstance(axs, np.ndarray) else [axs]

    top_labels, bottom_labels = [], []

    # 픽셀 격자 판별: 그리드=(이미지 폭, 높이)면 픽셀 격자로 간주
    px_grid_key = (int(round(img_w)), int(round(img_h)))

    for ax, (gw, gh) in zip(axs, grids[:ncols]):
        ax.set_xlim(0, img_w); ax.set_ylim(0, img_h)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

        cell_w, cell_h = img_w / gw, img_h / gh

        if which_cell and len(which_cell) == 2:
            cx_i, cy_i = int(which_cell[0]), int(which_cell[1])
        else:
            cx_i, cy_i = gw // 2, gh // 2

        cell_x0, cell_y0 = cx_i * cell_w, cy_i * cell_h

        cx_norm, cy_norm = float(norm_center[0]), float(norm_center[1])
        box_cx = cell_x0 + cx_norm * cell_w
        box_cy = cell_y0 + cy_norm * cell_h

        bw = float(norm_wh_cell[0]) * cell_w
        bh = float(norm_wh_cell[1]) * cell_h
        if anchor_rel_cell and len(anchor_rel_cell) == 2:
            bw *= float(anchor_rel_cell[0]); bh *= float(anchor_rel_cell[1])

        # 그리드 라인
        for i in range(gw + 1):
            ax.axvline(i * cell_w, color="#cfcfcf", lw=grid_lw, zorder=0)
        for j in range(gh + 1):
            ax.axhline(j * cell_h, color="#cfcfcf", lw=grid_lw, zorder=0)

        # 강조 셀
        ax.add_patch(Rectangle((cell_x0, cell_y0), cell_w, cell_h,
                            fill=False, edgecolor="#222", lw=1.6))
        # 박스
        ax.add_patch(Rectangle((box_cx - bw/2, box_cy - bh/2), bw, bh,
                            facecolor="#f0a63a", alpha=0.45,
                            edgecolor="#945c13", lw=1.6))

        # 패널 외곽 캡션 좌표
        bbox = ax.get_position(fig)
        cx = bbox.x0 + bbox.width/2

        is_px_grid = (gw, gh) == px_grid_key
        top_labels.append((cx, bbox.y1 + caption_top_pad,
                        f"{gw}×{gh} grid" + (" (pixels)" if is_px_grid else "")))
        if is_px_grid:
            # 픽셀 좌표(중심)
            px_x = int(round(box_cx)); px_y = int(round(box_cy))
            bottom_labels.append((cx, bbox.y0 - caption_bottom_pad,
                                f"px=({px_x},{px_y})  size≈({bw:.1f}px, {bh:.1f}px)"))
        else:
            bottom_labels.append((cx, bbox.y0 - caption_bottom_pad,
                                f"cell={cx_i},{cy_i}  size≈({bw:.1f}px, {bh:.1f}px)"))

    for x, y, s in top_labels:
        fig.text(x, y, s, ha="center", va="bottom", fontsize=12)
    for x, y, s in bottom_labels:
        fig.text(x, y, s, ha="center", va="top", fontsize=11, color="#333")

    fig.text(0.5, title_y, title, ha="center", va="top", fontsize=18)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

register(Grammar(
    "cell_scale",
    [],  # required 없음
    ["img_wh", "grids", "cell_xy", "norm_center", "norm_wh_cell",
    "anchor_rel_cell", "zoom", "dpi", "grid_lw", "title_pad", "title",
    "caption_top_pad", "caption_bottom_pad", "title_y"],
    render_cell_scale
))