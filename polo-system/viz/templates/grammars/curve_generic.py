# 범용 선 그래프(학습곡선, ROC/PR 등). seaborn 미사용, 색상 미지정.
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_curve_generic(inputs, out_path):
    series = inputs.get("series", [])
    if not series:
        return

    plt.figure(figsize=(5.2, 4.0))

    # 옵션
    style          = inputs.get("style", "line")      # "line" | "step"
    legend_loc     = inputs.get("legend_loc", None)   # 예: "upper right"
    xlim           = inputs.get("xlim", None)         # [xmin, xmax]
    ylim           = inputs.get("ylim", None)         # [ymin, ymax]
    annotate_last  = bool(inputs.get("annotate_last", False))
    diag           = bool(inputs.get("diag", False))  # ROC 기준선

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

        # 관측 누락 지점(임퓨트) 표시: observed_mask=false인 위치를 속빈 마커로
        mask = s.get("observed_mask")
        if isinstance(mask, list) and len(mask) == len(x):
            color = line.get_color()
            xi = [x[i] for i, obs in enumerate(mask) if not obs]
            yi = [y[i] for i, obs in enumerate(mask) if not obs]
            if xi:
                plt.scatter(xi, yi, s=46, facecolors="none", edgecolors=color, zorder=3)

        if annotate_last:
            plt.text(x[-1], y[-1], f"{y[-1]:.3g}", ha="left", va="bottom", fontsize=9)

    if any(s.get("label") for s in series):
        plt.legend(loc=legend_loc if legend_loc else None)

    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    if diag:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # ROC 참고선

    plt.xlabel(inputs.get("xlabel", ""))
    plt.ylabel(inputs.get("ylabel", ""))
    plt.title(inputs.get("title", ""))
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("curve_generic", ["series"], ["xlabel","ylabel","title",
                                              "style","legend_loc","xlim","ylim",
                                              "annotate_last","diag"], render_curve_generic))