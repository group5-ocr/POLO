# 누적 막대 차트 (오류 유형 분해, 리소스 비중, 데이터 구성 등)
import numpy as np
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_stack_bar(inputs, out_path):
    cats   = inputs.get("categories", [])
    series = inputs.get("series", [])
    title  = inputs.get("title","")
    ylabel = inputs.get("ylabel","")
    norm   = bool(inputs.get("normalize", False))
    annotate_top = bool(inputs.get("annotate", False))     # 상단 합계(정규화면 100%)
    rotate = int(inputs.get("rotate", 0))
    legend_out = inputs.get("legend_out", True)             # 범례를 밖으로
    inside_labels = bool(inputs.get("inside_labels", False))
    inside_threshold = float(inputs.get("inside_threshold", 7.0))  # % 기준

    if not cats or not series:
        return

    for s in series:
        v = s.get("values") or []
        if len(v) < len(cats):
            s["values"] = v + [0.0]*(len(cats)-len(v))

    data = np.vstack([s["values"] for s in series]).astype(float)  # (ns, nc)
    if norm:
        col_sum = data.sum(axis=0)
        col_sum[col_sum == 0.0] = 1.0
        data = 100.0 * data / col_sum

    x = np.arange(len(cats))
    right_margin = 0.85 if legend_out and len(series) > 1 else 1.0
    plt.figure(figsize=(max(6.4, 1.15*len(cats)), 4.0))
    ax = plt.gca()

    bottom = np.zeros(len(cats))
    bars_by_series = []
    for i, s in enumerate(series):
        bars = ax.bar(x, data[i], bottom=bottom, label=_lab(s["label"]))
        bars_by_series.append(bars)

        # 내부 퍼센트 라벨 (조각이 충분히 클 때만)
        if inside_labels and norm:
            for rect, val in zip(bars, data[i]):
                if val >= inside_threshold:
                    ax.text(rect.get_x()+rect.get_width()/2,
                            rect.get_y()+rect.get_height()/2,
                            f"{val:.0f}%", ha="center", va="center", fontsize=9)
        bottom += data[i]

    # 축/제목
    if title: ax.set_title(_lab(title))
    if ylabel: ax.set_ylabel(_lab(ylabel))
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=rotate)

    # 상단 합계 라벨 & 여백
    if annotate_top:
        if norm:
            ax.set_ylim(0, 108)  # 100% 라벨을 플롯 바깥으로 올리기 위한 여백
            for xi in x:
                ax.text(xi, 102, "100%", ha="center", va="bottom", fontsize=10)
        else:
            totals = bottom
            ymax = max(1.0, totals.max())
            ax.set_ylim(0, ymax*1.08)
            for xi, t in zip(x, totals):
                ax.text(xi, t*(1.003), f"{t:.3g}", ha="center", va="bottom", fontsize=9)

    if len(series) > 1:
        if legend_out:
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.legend(frameon=False, ncol=min(len(series),3))

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(rect=[0,0, right_margin, 1])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _lab(x):
    if isinstance(x, dict):
        return x.get("ko") or x.get("en") or ""
    return str(x)

register(Grammar("stack_bar",
                 ["categories","series"],
                 ["title","ylabel","normalize","annotate","rotate",
                  "legend_out","inside_labels","inside_threshold"],
                 render_stack_bar))