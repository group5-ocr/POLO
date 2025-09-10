# src/texprep/tex/chunk.py
# -*- coding: utf-8 -*-
"""
청킹 모듈 (modern typing)
- 문단 기준으로 자르고, 필요시 문장 분할로 과대 문단 쪼갬
- 섹션 경계(\section, \subsection, ...) 가급적 보존
- 청크 간 문단 overlap 지원
- 본문 내 플레이스홀더(⟨EQ⟩/⟨eq⟩/⟨FIG⟩/⟨TAB⟩) 카운트/리스트 제공
- JSONL로 덤프하기 좋은 dict 리스트 반환
"""

from __future__ import annotations
import re
from typing import Any

# ---- 분할 유틸 ----

_PARA_SPLIT = re.compile(r"\n\s*\n", re.S)
_SENT_SPLIT = re.compile(r"(?<!\b(?:e|i)\.g)(?<!\bvs)\.(?:\s+|$)|[!?](?:\s+|$)")

# LaTeX 섹션 탐지: \section{...}, \subsection{...}, \paragraph{...}
_SECTION_CMD = re.compile(
    r"^(?P<cmd>\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?)\s*\{(?P<title>[^}]*)\}\s*$"
)

# 플레이스홀더
# 닫힘 문자를 '>'가 아니라 '⟩' 기준으로 잡아야 안전
PH_EQ  = re.compile(r"⟨EQ:(?P<id>[^⟩]+)⟩")
PH_ieq = re.compile(r"⟨eq:(?P<id>[^⟩]+)⟩")
PH_FIG = re.compile(r"⟨FIG:(?P<id>[^⟩]+)⟩")
PH_TAB = re.compile(r"⟨TAB:(?P<id>[^⟩]+)⟩")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text) if p and p.strip()]


def _split_sentences(para: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(para) if s and s.strip()]
    return parts if parts else [para]


def _is_section_line(para: str) -> bool:
    return bool(_SECTION_CMD.match(para.strip()))


def _placeholder_stats(s: str) -> dict[str, Any]:
    eq_ids  = [m.group("id") for m in PH_EQ.finditer(s)]
    ieq_ids = [m.group("id") for m in PH_ieq.finditer(s)]
    fig_ids = [m.group("id") for m in PH_FIG.finditer(s)]
    tab_ids = [m.group("id") for m in PH_TAB.finditer(s)]
    return {
        "count": {
            "eq_display": len(eq_ids),
            "eq_inline": len(ieq_ids),
            "figure": len(fig_ids),
            "table": len(tab_ids),
        },
        "ids": {
            "eq_display": eq_ids,
            "eq_inline": ieq_ids,
            "figure": fig_ids,
            "table": tab_ids,
        },
    }


def _pack_paragraphs(
    paras: list[str],
    start: int,
    max_chars: int,
) -> tuple[int, str]:
    """
    start부터 최대 max_chars를 넘지 않는 선에서 문단을 합친다.
    단, 첫 문단 하나가 너무 길면 문장 분할로 쪼개서라도 한 덩어리 만든다.
    반환: (종료 다음 인덱스, 합쳐진 텍스트)
    """
    buf: list[str] = []
    size = 0
    i = start
    # 문단을 가능한 한 많이 채움
    while i < len(paras):
        p = paras[i]
        if size == 0 and len(p) > max_chars:
            # 첫 문단 자체가 과대. 문장 분할로 잘라 일부만 사용.
            lines: list[str] = []
            cur = 0
            for s in _split_sentences(p):
                if cur + len(s) + (1 if cur else 0) > max_chars:
                    break
                lines.append(s)
                cur += len(s) + (1 if cur else 0)
            # 문장 하나도 못 넣을 정도로 길면 그냥 하드 컷
            chunk_text = " ".join(lines) if lines else p[:max_chars]
            return i + 1, chunk_text
        # 다음 문단을 추가해도 제한 내면 이어붙임
        add_len = len(p) + (2 if size else 0)  # 문단 사이에 빈 줄 1개
        if size + add_len <= max_chars:
            buf.append(p)
            size += add_len
            i += 1
        else:
            break
    return i, ("\n\n".join(buf) if buf else paras[start][:max_chars])


def chunk_paragraphs(
    text: str,
    max_chars: int = 3800,
    overlap_paras: int = 1,
    prefer_section_boundary: bool = True,
) -> list[dict[str, Any]]:
    """
    문단 기반 청킹.
    - prefer_section_boundary: 청크 사이 경계를 섹션 줄에서 끊을 기회가 있으면 끊는다.
    - overlap_paras: 다음 청크 시작을 이전 청크 끝에서 몇 문단 겹칠지
    반환: [{chunk_index,start_para,end_para,text,char_count,placeholder:{count,ids}}...]
    """
    paras = _split_paragraphs(text)
    chunks: list[dict[str, Any]] = []
    i = 0
    cidx = 0

    while i < len(paras):
        # 섹션 경계 고려: 현재 i가 섹션줄이면 그 줄만 먼저 먹고 다음으로
        if prefer_section_boundary and _is_section_line(paras[i]):
            end_i, chunk_text = _pack_paragraphs(paras, i, max_chars)
            # 섹션 라인만 들어가면 다음 문단에서 합쳐도 됨
            # 다만 end_i == i+1일 확률 높음
        else:
            end_i, chunk_text = _pack_paragraphs(paras, i, max_chars)

        stats = _placeholder_stats(chunk_text)
        chunks.append({
            "chunk_index": cidx,
            "start_para": i,
            "end_para": end_i - 1,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "placeholder": stats,
        })
        cidx += 1

        if end_i >= len(paras):
            break

        # overlap 적용
        j = max(i + 1, end_i - max(0, overlap_paras))
        i = j

    return chunks


def chunk_sentences(
    text: str,
    max_chars: int = 3800,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    """
    문장 기반 대안 청킹(과학기술 논문에서 문단이 과도하게 길 때 사용).
    단락별로 문장 묶음 생성 후 이어붙인다.
    """
    paras = _split_paragraphs(text)
    # 문장화
    sents: list[str] = []
    for p in paras:
        sents.extend(_split_sentences(p))

    chunks: list[dict[str, Any]] = []
    cidx = 0
    i = 0
    while i < len(sents):
        buf: list[str] = []
        size = 0
        j = i
        while j < len(sents) and size + len(sents[j]) + (1 if size else 0) <= max_chars:
            buf.append(sents[j])
            size += len(sents[j]) + (1 if size else 0)
            j += 1
        if not buf:  # 한 문장도 못 넣는 극단적 케이스
            buf = [sents[i][:max_chars]]
            j = i + 1
        text_chunk = " ".join(buf)
        stats = _placeholder_stats(text_chunk)
        chunks.append({
            "chunk_index": cidx,
            "start_sent": i,
            "end_sent": j - 1,
            "text": text_chunk,
            "char_count": len(text_chunk),
            "placeholder": stats,
        })
        cidx += 1
        if j >= len(sents):
            break
        i = max(i + 1, j - max(0, overlap_sentences))
    return chunks


def choose_chunking(
    text: str,
    max_chars: int = 3800,
    overlap: int = 1,
    mode: str = "paragraph",  # "paragraph" | "sentence"
    prefer_section_boundary: bool = True,
) -> list[dict[str, Any]]:
    """
    라우팅 헬퍼.
    - mode="paragraph": 일반 추천
    - mode="sentence": 문단이 터무니없이 길거나 마크업 잔재가 많을 때
    """
    if mode == "sentence":
        return chunk_sentences(text, max_chars=max_chars, overlap_sentences=overlap)
    return chunk_paragraphs(
        text,
        max_chars=max_chars,
        overlap_paras=overlap,
        prefer_section_boundary=prefer_section_boundary,
    )
