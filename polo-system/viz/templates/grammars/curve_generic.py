# 범용 선 그래프(학습곡선, ROC/PR/Threshold/Focal/mAP/Calibration 등)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_curve_generic(inputs, out_path):
    series = inputs.get("series", [])
    if not series:
        return

    plt.figure(figsize=(5.2, 4.0))

    # 옵션
    style         = inputs.get("style", "line")      # "line" | "step"
    legend_loc    = inputs.get("legend_loc", None)   # 예: "upper right"
    xlim          = inputs.get("xlim", None)         # [xmin, xmax]
    ylim          = inputs.get("ylim", None)         # [ymin, ymax]
    annotate_last = bool(inputs.get("annotate_last", False))
    diag          = bool(inputs.get("diag", False))  # ROC 기준선
    kind          = inputs.get("kind", None)         # "threshold_sweep" | "focal_vs_ce" | "map_vs_iou" | "calibration"

    # 곡선
    for s in series:
        x = [float(v) for v in s.get("x", [])]
        y = [float(v) for v in s.get("y", [])]
        if len(x) < 2 or len(x) != len(y):
            continue

        label = s.get("label", None)
        if style == "step":
            line = plt.step(x, y, where="post", label=label)[0]
        else:
            line = plt.plot(x, y, marker="o", label=label)[0]

        # 관측 누락 지점(임퓨트) 표시: observed_mask=false → 속빈 마커
        mask = s.get("observed_mask")
        if isinstance(mask, list) and len(mask) == len(x):
            color = line.get_color()
            xi = [x[i] for i, obs in enumerate(mask) if not obs]
            yi = [y[i] for i, obs in enumerate(mask) if not obs]
            if xi:
                plt.scatter(xi, yi, s=46, facecolors="none", edgecolors=color, zorder=3)

        if annotate_last:
            plt.text(x[-1], y[-1], f"{y[-1]:.3g}", ha="left", va="bottom", fontsize=9)

    # 범례/limit/기준선
    if any(s.get("label") for s in series):
        plt.legend(loc=legend_loc if legend_loc else None)
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    if diag:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # ROC 참고선

    # 레이블/그리드
    plt.xlabel(inputs.get("xlabel", ""))
    plt.ylabel(inputs.get("ylabel", ""))
    plt.title(inputs.get("title", ""))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # 캡션(한 번만) - glossary에서 넘어온 플래그 기반
    msgs = []
    if diag:
        msgs.append("대각선=무작위 기준선(ROC)")
    if isinstance(series, list) and len(series) > 1:
        msgs.append("여러 모델 비교: 범례 참고")
    if annotate_last:
        msgs.append("마지막 지점 값을 표시")

    if kind == "threshold_sweep":
        msgs.append("임계↑ → Precision↑, Recall↓ · F1 최대점이 최적 임계 후보")
    elif kind == "focal_vs_ce":
        msgs.append("γ↑ → 쉬운 샘플 가중 감소, 어려운 샘플 비중↑")
    elif kind == "map_vs_iou":
        msgs.append("IoU 임계↑ → 더 엄격한 매칭 → mAP 감소 경향")
    elif kind == "calibration":
        msgs.append("완벽 보정: y=x · 아래쪽=과신, 위쪽=과소신")

    bottom = float(inputs.get("caption_bottom", 0.10))
    y_pos  = float(inputs.get("caption_y", 0.005))
    if msgs:
        plt.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
        plt.figtext(0.5, y_pos, "  ·  ".join(msgs), ha="center", va="bottom", fontsize=9)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


register(Grammar(
    "curve_generic",
    ["series"],  # required
    # optional
    ["xlabel", "ylabel", "title",
     "style", "legend_loc", "xlim", "ylim",
     "annotate_last", "diag",
     "kind", "caption_bottom", "caption_y"],
    render_curve_generic
))