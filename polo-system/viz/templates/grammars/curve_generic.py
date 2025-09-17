# 범용 선 그래프(학습곡선, ROC/PR/Threshold/Focal/mAP/Calibration 등)
import math
import matplotlib.pyplot as plt
from registry import Grammar, register


def _is_dummy_learning_series(series: list) -> bool:
    if not isinstance(series, list) or len(series) != 2:
        return False
    target_x = [1, 2, 3, 4, 5]
    y1 = [0.9, 0.7, 0.6, 0.5, 0.45]
    y2 = [1.0, 0.8, 0.7, 0.62, 0.60]
    def _eq(a, b):
        if len(a) != len(b): return False
        for p, q in zip(a, b):
            if isinstance(p, (int, float)) and isinstance(q, (int, float)):
                if abs(float(p) - float(q)) > 1e-9: return False
            else:
                if str(p) != str(q): return False
        return True
    s0, s1 = series[0], series[1]
    l0 = (s0.get("label","") or "").lower()
    l1 = (s1.get("label","") or "").lower()
    labels_ok = {l0, l1} == {"train", "val"}
    x_ok = _eq(s0.get("x", []), target_x) and _eq(s1.get("x", []), target_x)
    y_ok = (_eq(s0.get("y", []), y1) and _eq(s1.get("y", []), y2)) or (_eq(s0.get("y", []), y2) and _eq(s1.get("y", []), y1))
    return labels_ok and x_ok and y_ok


def render_curve_generic(inputs, out_path):
    series = inputs.get("series", [])
    if not series or _is_dummy_learning_series(series):
        # 숫자 없는 더미 스펙은 아예 렌더하지 않음.
        return

    plt.figure(figsize=(5.2, 4.0))

    # 옵션
    style         = inputs.get("style", "line")      # line, step
    legend_loc    = inputs.get("legend_loc", None)   # upper right
    xlim          = inputs.get("xlim", None)         # [xmin, xmax]
    ylim          = inputs.get("ylim", None)         # [ymin, ymax]
    annotate_last = bool(inputs.get("annotate_last", False))
    diag          = bool(inputs.get("diag", False))  # ROC 기준선
    kind          = inputs.get("kind", None)         # threshold_sweep

    # 곡선
    labels_seen = []
    for s in series:
        x = [float(v) for v in s.get("x", [])]
        y = [float(v) for v in s.get("y", [])]
        if len(x) < 2 or len(x) != len(y):
            continue

        label = s.get("label", None)
        if label: labels_seen.append(str(label).lower())

        if style == "step":
            line = plt.step(x, y, where="post", label=label)[0]
        else:
            line = plt.plot(x, y, marker="o", label=label)[0]

        # 관측 누락 지점 표시: observed_mask=false → 속빈 마커
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
    xlabel = str(inputs.get("xlabel", ""))
    plt.xlabel(xlabel)
    plt.ylabel(inputs.get("ylabel", ""))
    plt.title(inputs.get("title", ""))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # 캡션 — glossary 플래그 + 컨텍스트 가드
    msgs = []
    if diag: msgs.append("대각선=무작위 기준선(ROC)")
    lv = set(labels_seen)
    if len(series) > 1:
        if lv == {"train", "val"}:
            msgs.append("학습/검증 곡선 비교")
        else:
            msgs.append("여러 시리즈 비교: 범례 참고")
    if annotate_last: msgs.append("마지막 지점 값을 표시")

    xl = xlabel.lower()
    is_threshold_axis = ("threshold" in xl) or ("임계" in xl) or ("임계값" in xl) or ("임계치" in xl)
    if kind == "threshold_sweep" and is_threshold_axis:
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
    ["series"],
    ["xlabel","ylabel","title","style","legend_loc","xlim","ylim",
    "annotate_last","diag","kind","caption_bottom","caption_y"],
    render_curve_generic
))