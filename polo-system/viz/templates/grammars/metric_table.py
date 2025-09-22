import matplotlib.pyplot as plt
from matplotlib.table import Table
from registry import Grammar, register

def _label(v):
    if isinstance(v, dict):
        return v.get("ko") or v.get("en") or ""
    return str(v)

def render_metric_table(inputs, out_path):
    # 필수
    methods = inputs.get("methods", [])
    metrics = inputs.get("metrics", [])
    values  = inputs.get("values", [])

    # 제목/캡션
    title   = _label(inputs.get("title", {"ko": "평가지표 해석"}))
    caption = inputs.get("caption", "")

    # 레이아웃/스타일 파라미터 (입력으로 오버라이드 가능)
    fig_w, fig_h = float(inputs.get("fig_w", 7.0)), float(inputs.get("fig_h", 5.0))
    title_size = int(inputs.get("title_size", 24))
    cell_size = int(inputs.get("cell_size", 16))
    caption_size = int(inputs.get("caption_size", 13))

    # 표 내부 세로 패딩(행 높이 배율).
    row_pad = float(inputs.get("row_pad", 1.00))

    # 제목/캡션 위치(figure 좌표)
    # title_y는 0.90~0.95, caption_y는 0.025~0.06 권장
    title_y = float(inputs.get("title_y", 0.92))
    title_gap        = float(inputs.get("title_gap", 0.030))
    # 표-캡션 사이 간격(작게 줄일수록 캡션이 위로 붙음)
    caption_gap      = float(inputs.get("caption_gap", 0.020))

    bottom_margin = 0.08

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.98, bottom=0.06)
    ax = fig.add_subplot(111)
    ax.axis("off")

    # 제목
    if title:
        fig.text(0.5, title_y, title, ha="center", va="bottom", fontsize=title_size)

    # 표 그리기
    nrows = len(methods) + 1
    ncols = len(metrics) + 1

    # 표가 차지할 박스(figure 내 비율).
    top_edge = title_y - title_gap
    tbl_bbox = [0.08, bottom_margin, 0.84, top_edge - bottom_margin]
    tbl = Table(ax, bbox=tbl_bbox)

    # 셀 기본 크기(상대적 세팅).
    cw = 1.0 / ncols
    ch = 0.9 / nrows

    # 좌측 헤더(행 이름) 레이블
    row_header = _label(inputs.get("row_header", {"ko": "등급", "en": "Level"}))
    tbl.add_cell(0, 0, cw, ch, text=row_header, loc="center",
                facecolor="white", edgecolor="black")

    # 상단 컬럼 헤더
    for j, m in enumerate(metrics):
        tbl.add_cell(0, j + 1, cw, ch, text=_label(m), loc="center",
                    facecolor="white", edgecolor="black")

    # 본문
    for i, r in enumerate(methods):
        # 행 이름
        tbl.add_cell(i + 1, 0, cw, ch, text=_label(r), loc="center",
                    facecolor="white", edgecolor="black")
        # 값
        for j in range(len(metrics)):
            txt = ""
            if i < len(values) and j < len(values[i]):
                v = values[i][j]
                if isinstance(v, (int, float)):
                    txt = f"{v:g}"
                else:
                    txt = str(v)
            tbl.add_cell(i + 1, j + 1, cw, ch, text=txt, loc="center",
                        facecolor="white", edgecolor="black")

    # 폰트/셀 높이 조정
    for (_, _), cell in tbl.get_celld().items():
        cell.get_text().set_fontsize(cell_size)

    ax.add_table(tbl)

    # 캡션 (figure 좌표로 직접 배치 → 값만 바꾸면 즉시 반영)
    if caption:
        fig.canvas.draw()
        tbl_box = tbl.get_window_extent(fig.canvas.get_renderer())\
                    .transformed(fig.transFigure.inverted())
        cap_y = max(0.02, tbl_box.y0 - caption_gap)
        fig.text(0.5, cap_y, caption, ha="center", va="top", fontsize=caption_size)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# Grammar 등록 (포지셔널 인자 사용: name, needs, optionals, renderer)
register(Grammar(
    "metric_table",
    ["methods", "metrics", "values"],
    ["title", "caption",
    "fig_w", "fig_h",
    "title_size", "cell_size", "caption_size",
    "row_pad", "title_y", "caption_y", "row_header"],
    render_metric_table
))