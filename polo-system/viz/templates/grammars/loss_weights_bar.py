# YOLO 손실 가중치 막대그래프
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_loss_weights_bar(inputs, out_path):
    ws = inputs.get("weights", {}) or {}
    if not ws:  # 값 없으면 그리지 않음
        return

    labels = list(ws.keys())
    values = [float(ws[k]) for k in labels]

    plt.figure(figsize=(6.4, 4.0), dpi=200)
    ax = plt.gca()
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color="#f0a63a", edgecolor="#945c13")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("weight", fontsize=11)
    ax.set_title(inputs.get("title", "YOLO 손실 가중치"), fontsize=14)

    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.05, yi, f"{v:g}", va="center", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

register(Grammar(
    "loss_weights_bar",
    [],                        # required 없음
    ["weights", "title"],      # optional 인자
    render_loss_weights_bar    # 렌더러 함수
))