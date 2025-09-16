# 히스토그램 문법
import matplotlib.pyplot as plt
import statistics
from registry import Grammar, register

def render_histogram(inputs, out_path):
    values = inputs.get("values") or []
    if not values or len(values) < 20:
        return  # 데이터가 너무 적으면 그리지 않음
    if len(set(values)) < 8:
        return  # 유니크 값이 너무 적으면 무시
    try:
        if statistics.pstdev(values) == 0:
            return  # 모두 같은 값이면 히스토그램 무의미
    except Exception:
        return

    bins = inputs.get("bins", "fd")  # "fd"|"sturges"|int
    plt.figure(figsize=(5.6, 4.0))
    plt.hist(values, bins=bins if isinstance(bins, (int,str)) else 20)
    plt.xlabel(inputs.get("xlabel", "value"))
    plt.ylabel(inputs.get("ylabel", "Count"))
    plt.title(inputs.get("title", "Histogram"))
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar(
    "histogram",
    ["values"],
    ["bins", "xlabel", "ylabel", "title"],
    render_histogram
))
