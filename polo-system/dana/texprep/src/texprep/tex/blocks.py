# src/texprep/tex/blocks.py
# -*- coding: utf-8 -*-
"""
figure/table 블록 추출기 (modern typing)
- \begin{figure[*]}...\end{figure[*]}, \begin{table[*]}...\end{table[*]} 모두 지원
- \caption{...}, \label{...}, \includegraphics[...]{} 추출
- subfigure/subtable 안의 \subcaption, \label, \includegraphics 수집
- \captionof{figure|table}{...} 패턴도 best-effort로 잡아 'pseudo' 자산 생성
- 본문 텍스트에는 ⟨FIG:label⟩ / ⟨TAB:label⟩ 플레이스홀더로 치환
"""

from __future__ import annotations
import re
from typing import Any

# 공백 정규화
def _nw(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# 캡션, 라벨, 그래픽스 파서
_CAPTION_RE = re.compile(r"\\caption(?:\[[^\]]*\])?\{(?P<cap>.*?)\}", re.S)
_LABEL_RE   = re.compile(r"\\label\{(?P<label>[^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(?P<path>[^}]+)\}")

# subfigure/subtable 패턴
_SUBENV_RE = re.compile(
    r"\\begin\{(?P<kind>subfigure|subtable)\}.*?\\end\{(?P=kind)\}",
    re.S
)
_SUBCAPTION_RE = re.compile(r"\\subcaption(?:\[[^\]]*\])?\{(?P<cap>.*?)\}", re.S)

# figure/table(별표 포함) 블록 매치
_FLOAT_RE = re.compile(
    r"\\begin\{(?P<env>figure\*?|table\*?)\}(?P<body>.*?)\\end\{(?P=env)\}",
    re.S
)

# \captionof{figure}{...} 같은 '떠있는' 캡션
_CAPTIONOF_RE = re.compile(
    r"\\captionof\{(?P<kind>figure|table)\}\{(?P<cap>.*?)\}",
    re.S
)

def _parse_subitems(block: str) -> list[dict[str, Any]]:
    """subfigure/subtable 블록 내부의 자식 자산 파싱."""
    items: list[dict[str, Any]] = []
    for m in _SUBENV_RE.finditer(block):
        raw = m.group(0)
        cap_m = _SUBCAPTION_RE.search(raw)
        label_m = _LABEL_RE.search(raw)
        graphics = [g.group("path") for g in _GRAPHICS_RE.finditer(raw)]
        items.append({
            "kind": m.group("kind"),
            "label": label_m.group("label") if label_m else None,
            "caption": _nw(cap_m.group("cap")) if cap_m else "",
            "graphics": graphics,
        })
    return items

def _asset_id(prefix: str, idx: int, label: str | None) -> str:
    return label if label else f"{prefix}_{idx}"

def extract_assets(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    figure/table 자산을 추출하고 본문을 플레이스홀더로 치환한다.
    반환:
      - new_text: 자산 블록이 ⟨FIG:...⟩ / ⟨TAB:...⟩로 치환된 텍스트
      - assets: [{id, kind, env, label, caption, graphics, children, raw?}]
    """
    assets: list[dict[str, Any]] = []
    idx_fig = 0
    idx_tab = 0

    def repl(m: re.Match) -> str:
        nonlocal idx_fig, idx_tab
        env = m.group("env")            # figure, figure*, table, table*
        body = m.group("body")
        is_figure = env.startswith("figure")

        # 기본 메타
        cap_m = _CAPTION_RE.search(body)
        label_m = _LABEL_RE.search(body)
        graphics = [g.group("path") for g in _GRAPHICS_RE.finditer(body)]
        children = _parse_subitems(body)

        label = label_m.group("label") if label_m else None
        caption = _nw(cap_m.group("cap")) if cap_m else ""

        if is_figure:
            idx_fig += 1
            aid = _asset_id("fig", idx_fig, label)
            tag = f"⟨FIG:{aid}⟩"
            kind = "figure"
        else:
            idx_tab += 1
            aid = _asset_id("tab", idx_tab, label)
            tag = f"⟨TAB:{aid}⟩"
            kind = "table"

        assets.append({
            "id": aid,
            "kind": kind,           # figure | table
            "env": env,             # figure, figure*, table, table*
            "label": label,         # 원 라벨 (없을 수 있음)
            "caption": caption,
            "graphics": graphics,   # 포함된 이미지 경로들
            "children": children,   # subfigure/subtable 목록
        })
        return tag

    new_text = _FLOAT_RE.sub(repl, text)

    # 캡션만 있는 \captionof{figure}{...} 처리: pseudo 자산
    def repl_captionof(m: re.Match) -> str:
        nonlocal idx_fig, idx_tab
        kind = m.group("kind")    # figure | table
        caption = _nw(m.group("cap"))
        if kind == "figure":
            idx_fig += 1
            aid = f"fig_{idx_fig}"
            tag = f"⟨FIG:{aid}⟩"
        else:
            idx_tab += 1
            aid = f"tab_{idx_tab}"
            tag = f"⟨TAB:{aid}⟩"
        assets.append({
            "id": aid,
            "kind": kind,
            "env": "captionof",
            "label": None,
            "caption": caption,
            "graphics": [],
            "children": [],
        })
        return tag

    new_text = _CAPTIONOF_RE.sub(repl_captionof, new_text)

    return new_text, assets
