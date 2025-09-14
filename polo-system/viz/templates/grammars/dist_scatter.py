# 산점도 문법 (예: (w,h) 분포)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_dist_scatter(inputs, out_path):
    points = inputs.get("points", [])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.figure(figsize=(5.2,5.0))
    plt.scatter(xs, ys, s=10, alpha=0.7)
    plt.xlabel(inputs.get("xlabel", "x"))
    plt.ylabel(inputs.get("ylabel", "y"))
    plt.title(inputs.get("title",""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("dist_scatter", ["points"], ["xlabel","ylabel","title"], render_dist_scatter))
