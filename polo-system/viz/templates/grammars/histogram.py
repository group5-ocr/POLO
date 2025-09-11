# 히스토그램 문법
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_histogram(inputs, out_path):
    values = inputs.get("values", [])
    bins = inputs.get("bins", 20)
    plt.figure(figsize=(5.6, 4.0))
    plt.hist(values, bins=bins)
    plt.xlabel(inputs.get("xlabel",""))
    plt.ylabel("Count")
    plt.title(inputs.get("title",""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar("histogram", ["values"], ["bins","xlabel","title"], render_histogram))
