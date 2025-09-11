# 메트릭 비교 표 문법 (methods x metrics)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_metric_table(inputs, out_path):
    methods = inputs.get("methods", [])
    metrics = inputs.get("metrics", [])
    values  = inputs.get("values", [])
    n_rows = len(methods)
    n_cols = len(metrics)

    plt.figure(figsize=(max(4, n_cols*1.2), max(2.5, n_rows*0.6)))
    ax = plt.gca()
    ax.axis("off")
    celltext = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            try:
                row.append(f"{values[r][c]:.3f}")
            except Exception:
                row.append("")
        celltext.append(row)
    table = plt.table(cellText=celltext, rowLabels=methods, colLabels=metrics, loc="center")
    table.scale(1, 1.2)
    plt.title(inputs.get("title", "Metrics"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("metric_table", ["methods","metrics","values"], ["title"], render_metric_table))
