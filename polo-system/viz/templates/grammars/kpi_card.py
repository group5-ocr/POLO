# 핵심 지표(KPI) 카드 문법 (큰 숫자 강조)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_kpi_card(inputs, out_path):
    title = inputs.get("title", "KPI")
    value = inputs.get("value", "—")
    subtitle = inputs.get("subtitle", "")
    plt.figure(figsize=(4.5, 3.0))
    plt.axis("off")
    plt.text(0.5, 0.75, title, ha="center", va="center", fontsize=12)
    plt.text(0.5, 0.45, str(value), ha="center", va="center", fontsize=28)
    if subtitle:
        plt.text(0.5, 0.22, subtitle, ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("kpi_card", ["value"], ["title","subtitle"], render_kpi_card))
