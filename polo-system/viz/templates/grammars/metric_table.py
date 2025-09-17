import matplotlib.pyplot as plt
from registry import Grammar, register

def _t(v):
    return (v.get("ko") or v.get("en") or "") if isinstance(v, dict) else str(v)

def render_metric_table(inputs, out_path):
    title    = _t(inputs.get("title") or {"ko":"평가지표 해석"})
    headline = _t(inputs.get("headline")) if inputs.get("headline") else None
    caption  = _t(inputs.get("caption"))  if inputs.get("caption")  else None

    headers = inputs.get("headers")
    rows    = inputs.get("rows")
    # methods/metrics/values 스키마 지원
    if rows is None and {"methods","metrics","values"} <= set(inputs):
        methods = list(inputs["methods"])
        metrics = list(inputs["metrics"])
        values  = list(inputs["values"])
        headers = [inputs.get("first_col_label","등급")] + metrics
        rows    = [[methods[i]] + [values[i][j] for j in range(len(metrics))] for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=6)

    # 위쪽 headline
    top_limit = 0.92
    if headline:
        y_head = float(inputs.get("headline_y", 0.935))
        plt.figtext(0.5, y_head, headline, ha="center", va="top", fontsize=12)
        top_limit = min(top_limit, y_head - 0.02)

    # 숫자 포맷
    def f(x):
        if isinstance(x, (int, float)):
            return f"{x:.3f}".rstrip("0").rstrip(".")
        return str(x)
    rows_fmt = [[f(c) for c in r] for r in rows]

    tbl = ax.table(cellText=rows_fmt, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(float(inputs.get("cell_fontsize", 12)))
    tbl.scale(1.0, 1.2)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_height(cell.get_height()*1.15)
            cell.set_fontsize(float(inputs.get("header_fontsize", 12.5)))
            cell.set_edgecolor("black")
        else:
            cell.set_edgecolor("black")

    # 캡션 자동 생성(없을 때)
    if not caption:
        low = title.lower()
        if "map" in low:
            caption = "COCO mAP@[.5:.95]: IoU 0.50–0.95(0.05 간격)에서 AP 평균. 아래 등급은 최소 mAP 기준."
        elif "accuracy" in low or "정확도" in low:
            caption = "정확도=(TP+TN)/전체. 클래스 불균형 상황에선 보조 지표와 함께 해석."
        elif "f1" in low:
            caption = "F1=2·(Precision·Recall)/(Precision+Recall). 두 지표 균형을 요약."

    bottom = float(inputs.get("caption_bottom", 0.12))
    cap_y  = float(inputs.get("caption_y", 0.006))
    if caption:
        plt.tight_layout(rect=(0.0, bottom, 1.0, top_limit))
        plt.figtext(0.5, cap_y, caption, ha="center", va="bottom", fontsize=10)
    else:
        plt.tight_layout(rect=(0.0, 0.05, 1.0, top_limit))

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar(
    "metric_table",
    [],
    ["title","headers","rows","methods","metrics","values","first_col_label",
    "headline","headline_y","caption","caption_bottom","caption_y",
    "cell_fontsize","header_fontsize"],
    render_metric_table
))