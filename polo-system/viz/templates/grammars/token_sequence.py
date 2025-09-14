# 토큰 시퀀스 라인 (CLS/SEP 강조)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from registry import Grammar, register

def render_token_sequence(inputs, out_path):
    tokens = inputs.get("tokens", ["[CLS]","sentA","[SEP]","sentB","[SEP]"])
    notes  = inputs.get("notes", {"[CLS]":"doc cls","[SEP]":"boundary"})
    title  = inputs.get("title", "Token Sequence")

    n = max(1, len(tokens))
    plt.figure(figsize=(max(5.0, 1.0 + 0.9*n), 1.9))
    ax = plt.gca(); ax.axis("off")
    ax.set_xlim(0, n); ax.set_ylim(0, 1)

    box_w, box_h = 0.75, 0.36
    y_box = 0.55
    for i, t in enumerate(tokens):
        x = i + 0.125  # 좌우 여백
        ax.add_patch(Rectangle((x, y_box - box_h/2), box_w, box_h, fill=False))
        ax.text(x + box_w/2, y_box, t, ha="center", va="center")
        if t in notes:
            ax.text(x + box_w/2, 0.22, notes[t], ha="center", va="center", fontsize=9)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("token_sequence", ["tokens"], ["title","notes"], render_token_sequence))