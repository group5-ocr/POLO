# 그룹 막대 그래프 문법 (멀티 시리즈 + 범례 + 포맷 옵션 지원)
import numpy as np
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_bar_group(inputs, out_path):
    cats     = inputs.get("categories", [])
    series   = inputs.get("series", [])
    title    = inputs.get("title", "")
    ylabel   = inputs.get("ylabel", "")
    ylim     = inputs.get("ylim", None)
    annotate = bool(inputs.get("annotate", False))
    legend   = bool(inputs.get("legend", True))
    rotate   = int(inputs.get("rotate", 0))
    fmt      = inputs.get("fmt", "auto")
    legend_out = inputs.get("legend_out", True)

    if not series or not any(s.get("values") for s in series) or not cats:
        return

    for s in series:
        vals = (s.get("values") or [])
        if len(vals) < len(cats):
            s["values"] = vals + [0.0]*(len(cats)-len(vals))

    ns = len(series)
    x = np.arange(len(cats))
    width = 0.75 if ns == 1 else max(0.8/ns, 0.16)

    right_margin = 0.85 if legend_out and legend and ns>=2 else 1.0
    plt.figure(figsize=(max(6.4, 1.15*len(cats)), 4.2))
    ax = plt.gca()

    bar_groups = []
    if ns == 1:
        bars = ax.bar(x, series[0]["values"], width=0.75, label=_lab(series[0]["label"]))
        bar_groups.append(bars)
    else:
        for i, s in enumerate(series):
            offs = (i - ns/2 + 0.5)*width
            bars = ax.bar(x + offs, s["values"], width=width, label=_lab(s["label"]))
            bar_groups.append(bars)

    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=rotate)
    if ylabel: ax.set_ylabel(_lab(ylabel))
    if title:  ax.set_title(_lab(title))

    # y한계 자동 여백
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ymax = max([max(s["values"]) for s in series] + [1.0])
        ax.set_ylim(0, ymax*1.12)

    # 포맷
    ylab_text = _lab(ylabel) if ylabel else ""
    is_pct = "%" in ylab_text
    def _format(v):
        if fmt == "percent" or (fmt == "auto" and is_pct): return f"{v:.1f}%"
        if fmt.startswith("float:"): return f"{v:{fmt.split(':',1)[1]}}"
        if fmt == "int": return f"{int(round(v))}"
        return f"{v:.3g}"

    if annotate:
        if ns == 1:
            for rect, v in zip(bar_groups[0], series[0]["values"]):
                ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()*1.005,
                        _format(v), ha="center", va="bottom", fontsize=9)
        else:
            for bars, s in zip(bar_groups, series):
                for rect, v in zip(bars, s["values"]):
                    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()*1.005,
                            _format(v), ha="center", va="bottom", fontsize=8)

    if legend and ns >= 2:
        if legend_out:
            ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
        else:
            ax.legend(frameon=False, ncol=min(ns,3))

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(rect=[0,0, right_margin, 1])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _lab(x):
    if isinstance(x, dict): return x.get("ko") or x.get("en") or ""
    return str(x)

register(Grammar("bar_group",
                ["categories","series"],
                ["title","ylabel","ylim","annotate","legend","rotate","fmt","legend_out"],
                render_bar_group))