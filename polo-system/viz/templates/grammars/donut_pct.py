# 도넛형 퍼센트 차트 (여러 논문에서 구성 비율 표시에 유용)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_donut_pct(inputs, out_path):
    parts = inputs.get("parts", [])  # 예: [("MASK",80),("Random",10),("Keep",10)]
    labels = [p[0] for p in parts]
    values = [float(p[1]) for p in parts]
    if not values:
        values = [1.0]
        labels = ["N/A"]

    fig = plt.figure(figsize=(4.6, 3.6))
    ax = plt.gca()
    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.45), startangle=90)
    ax.legend(wedges, [f"{l} ({v:g}%)" for l, v in zip(labels, values)],
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set(aspect="equal")
    plt.title(inputs.get("title", "Composition"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("donut_pct", ["parts"], ["title"], render_donut_pct))