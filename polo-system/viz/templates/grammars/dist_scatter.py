# 산점도 문법 (예: (w,h) 분포)
import matplotlib.pyplot as plt
from registry import Grammar, register

def render_dist_scatter(inputs, out_path):
    # 실데이터 없으면 생성 안 함(예시/더미 차단)
    points = inputs.get("points", [])
    if not isinstance(points, list) or len(points) < 5:
        return
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except Exception:
        return
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return  # 한 점/수평/수직선 수준이면 무의미 → 생성 안 함

    # 라벨/제목/캡션 (사전/문자 모두 허용)
    def _lab(v):
        if isinstance(v, dict):
            return v.get("ko") or v.get("en") or ""
        return str(v)
    title   = _lab(inputs.get("title", ""))
    xlabel  = _lab(inputs.get("xlabel", "x"))
    ylabel  = _lab(inputs.get("ylabel", "y"))
    caption = _lab(inputs.get("caption", ""))

    # 하단 캡션 간격(프로젝트 공통 규격) — 필요시 호출 측에서 override
    caption_bottom = float(inputs.get("caption_bottom", 0.10))  # tight_layout rect 하한
    caption_y      = float(inputs.get("caption_y", 0.005))      # figtext y 위치(0~1)

    # 그리기
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(xs, ys, s=10, alpha=0.7)  # 색상/스타일 지정 안 함(전역 규칙 준수)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # 캡션이 있으면 겹침 방지하여 배치 (없으면 레이아웃 보정도 생략)
    if caption:
        plt.tight_layout(rect=(0.0, caption_bottom, 1.0, 1.0))
        plt.figtext(0.5, caption_y, caption, ha="center", va="bottom", fontsize=9)
    else:
        plt.tight_layout()

    # 저장
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

register(Grammar(
    "dist_scatter",
    ["points"],
    ["xlabel","ylabel","title","caption","caption_bottom","caption_y"],
    render_dist_scatter
))