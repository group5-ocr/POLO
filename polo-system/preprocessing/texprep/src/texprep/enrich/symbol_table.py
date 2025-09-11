# # -*- coding: utf-8 -*-
# """
# AI 논문 전용: 기호 사전/프리림 요약/수식 셀렉션 + Qwen 입력 페이로드 생성(메시지 포맷 없음)

# 이 모듈은 전처리된 본문과 수식 리스트를 받아서
# - symbol_table: 기호 정의 후보 랭크
# - prelim_summary: 표기/가정/프리림 문맥 요약
# - equations_selected: 메인 수식 상위 N개(라텍스 원형 유지)
# - qwen_payload: 위 자료를 한 덩어리(dict)로 반환 (role 없이)
# """

# from __future__ import annotations
# from typing import Any, Iterable
# import re
# from collections import Counter, defaultdict

# # -------- 유틸 --------
# _WS = re.compile(r"\s+")
# def _norm_space(s: str) -> str: return _WS.sub(" ", s).strip()
# def _split_paragraphs(text: str) -> list[str]:
#     return [p for p in re.split(r"\n\s*\n", text) if p.strip()]
# def _split_sentences(para: str) -> list[str]:
#     return [s.strip() for s in re.split(r"(?<!\b(?:e|i)\.g)(?<!\bvs)\.(?:\s+|$)|[!?](?:\s+|$)", para) if s.strip()]

# # -------- 연산자/포맷/알리아스 --------
# RELATION_CMDS = {
#     "in","sim","propto","approx","triangleq","le","ge","neq","cdot","times","nabla","partial"
# }
# CMD_TO_UNI = {
#     "in":"∈","sim":"∼","propto":"∝","approx":"≈","triangleq":"≜",
#     "le":"≤","ge":"≥","neq":"≠","cdot":"·","times":"×","nabla":"∇","partial":"∂"
# }
# OP_UNI = set(CMD_TO_UNI.values()) | {"∇","∂"}

# STOP_CMDS = {
#     "scriptsize", "normalsize", "text", "textrm", "textsc", "rm", "bf", "it",
#     "small", "large", "emph",  # 전부 포맷/텍스트
# }
# ALIASES = {
#     r"\Pr": "P",
#     r"\mathbb{P}": "P",
# }
# CMD_NAME_RE = re.compile(r"^\\([A-Za-z]+)")
# def _is_stopped_cmd(tok: str) -> bool:
#     m = CMD_NAME_RE.match(tok)
#     return bool(m and m.group(1) in STOP_CMDS)

# # -------- 기호 토큰 추출 --------
# CMD_TOKEN     = r"\\[A-Za-z]+(?:\s*\{[^{}]*\})?"
# GREEK_OR_WORD = r"[A-Za-z](?:_[A-Za-z0-9]+|\^{[^}]+}|\^[A-Za-z0-9])?"
# BB_CAL_BF     = r"\\(?:mathbb|mathcal|mathbf|boldsymbol|bm)\s*\{[A-Za-z]\}"
# SET_SYM       = r"[A-Z](?:_[A-Za-z0-9]+)?"
# OP_SYM        = r"[∇∂]"
# SPECIALS      = r"(?:\\mathbb\{[RNZEP]\}|\\mathrm\{KL\}|\\mathcal\{L\})"

# SYMBOL_RE = re.compile(rf"({SPECIALS}|{OP_SYM}|{BB_CAL_BF}|{CMD_TOKEN}|{GREEK_OR_WORD}|{SET_SYM})")

# NORM_WRAP_CMDS = re.compile(r"\\(?:mathbf|boldsymbol|bm|mathrm|mathit|text|operatorname)\s*\{([^{}]+)\}")
# NORM_MATHBB    = re.compile(r"\\mathbb\s*\{([A-Za-z])\}")
# NORM_MATHCAL   = re.compile(r"\\mathcal\s*\{([A-Za-z])\}")
# NORM_BRACED    = re.compile(r"^\{(.+)\}$")

# def normalize_symbol(sym: str) -> str:
#     s = sym.strip()
#     s = NORM_BRACED.sub(r"\1", s)
#     s = NORM_WRAP_CMDS.sub(r"\1", s)
#     s = NORM_MATHBB.sub(r"\1", s)
#     s = NORM_MATHCAL.sub(r"\1", s)
#     m = CMD_NAME_RE.match(s)
#     if m and "{" not in s:
#         name = m.group(1)
#         if name in RELATION_CMDS:
#             return CMD_TO_UNI.get(name, "\\" + name)  # 연산자는 보존/유니코드화
#         s = name
#     # 원본 토큰 기준 알리아스
#     if sym in ALIASES:
#         s = ALIASES[sym]
#     return s.replace(" ", "")

# # -------- 정의/설명 문장 패턴 --------
# CUE_PHRASES = (
#     r"(?:[Ww]e\s+define)", r"(?:[Ww]e\s+denote)", r"(?:[Ww]e\s+use)",
#     r"(?:[Ll]et\b)", r"(?:[Ww]here\b)", r"(?:[Hh]ere,|\bthis\s+denotes\b)",
#     r"(?:[Aa]ssume\b|[Ww]e\s+assume)", r"(?:[Pp]reliminaries\b|[Nn]otation\b)",
# )
# CUES_RE = re.compile("|".join(CUE_PHRASES))

# REL_PATTERNS = (
#     r"\\in", r"\\sim", r"=", r"\\propto", r"\\approx", r"\\triangleq",
#     r"\\mathbb\{[RZNEP]\}", r"\\mathrm\{KL\}", r"\\mathcal\{L\}", r"\\nabla", r"\\partial",
#     r"∈", r"∼", r"∝", r"≈", r"≜", r"≤", r"≥", r"≠", r"·", r"×", r"∇", r"∂"
# )

# # -------- 심볼/정의 빌드 --------
# def extract_symbols_from_equation(eq_tex: str, max_symbols: int = 50) -> list[str]:
#     syms: list[str] = []
#     for m in SYMBOL_RE.finditer(eq_tex):
#         tok = m.group(0)
#         if tok.startswith("\\begin") or tok.startswith("\\end"):
#             continue
#         if _is_stopped_cmd(tok):
#             continue
#         n = normalize_symbol(tok)
#         if not n or len(n) > 20:
#             continue
#         # 전부 대문자 3자 이상(예: IOU, CLASS) → 라벨로 보고 제외
#         if n.isupper() and len(n) >= 3:
#             continue
#         if n not in syms:
#             syms.append(n)
#         if len(syms) >= max_symbols:
#             break
#     return syms

# def _sent_score_for_symbol(sent: str, sym: str) -> float:
#     score = 0.0
#     if CUES_RE.search(sent): score += 2.0
#     for pat in REL_PATTERNS:
#         if re.search(pat, sent): score += 0.6
#     if sym in sent: score += 1.0
#     L = len(sent)
#     if L > 220: score -= 0.5
#     if L > 320: score -= 0.8
#     return score

# EQ_MARKER_RE = re.compile(r"⟨EQ:(eq_\d+)⟩")

# def _eq_positions_by_para(paragraphs: list[str]) -> dict[str, list[int]]:
#     pos = defaultdict(list)
#     for p_idx, p in enumerate(paragraphs):
#         for eq_id in EQ_MARKER_RE.findall(p):
#             pos[eq_id].append(p_idx)
#     return pos

# # 단문자 라틴 변수 경계 매칭 (i, j, x 같은 오탐 방지)
# def _has_boundary_match(txt: str, sym: str) -> bool:
#     if len(sym) == 1 and sym.isalpha() and sym.isascii():
#         return bool(re.search(rf"(?<![A-Za-z]){re.escape(sym)}(?![A-Za-z])", txt))
#     return sym in txt

# def build_symbol_table(
#     paragraphs: list[str],
#     equations: list[dict[str, Any]],
#     window_sentences: int = 2,
#     max_defs_per_symbol: int = 2,
# ) -> list[dict[str, Any]]:
#     sym_freq: Counter[str] = Counter()
#     sym_eqs: dict[str, list[str]] = defaultdict(list)
#     eq_map = {e.get("id"): e.get("tex", "") for e in equations}

#     for eq in equations:
#         eq_id = eq.get("id", "")
#         seen = set()
#         for s in extract_symbols_from_equation(eq.get("tex", "")):
#             if s in seen:
#                 continue
#             seen.add(s)
#             sym_freq[s] += 1
#             if eq_id:
#                 sym_eqs[s].append(eq_id)

#     # 방정식 → 문단 인덱스
#     eq_pos = _eq_positions_by_para(paragraphs)

#     symbol_defs: dict[str, list[dict[str, Any]]] = defaultdict(list)

#     for sym, eq_ids in sym_eqs.items():
#         # 이 심볼이 등장한 방정식이 있는 문단 주변만 후보
#         para_candidates: set[int] = set()
#         for eid in set(eq_ids):
#             for p_idx in eq_pos.get(eid, []):
#                 para_candidates.update({p_idx - 1, p_idx, p_idx + 1})
#         para_candidates = {i for i in para_candidates if 0 <= i < len(paragraphs)}

#         for p_idx in sorted(para_candidates):
#             sents = _split_sentences(paragraphs[p_idx])
#             for s_idx, _ in enumerate(sents):
#                 lo = max(0, s_idx - window_sentences)
#                 hi = min(len(sents), s_idx + window_sentences + 1)
#                 snippet = _norm_space(". ".join(sents[lo:hi]))
#                 if not snippet:
#                     continue

#                 # 단문자 라틴 변수는 경계 일치 없으면 컷
#                 if len(sym) == 1 and sym.isalpha() and sym.isascii():
#                     if not _has_boundary_match(snippet, sym):
#                         continue

#                 sc = _sent_score_for_symbol(snippet, sym)
#                 # 경계 일치 가점
#                 if len(sym) == 1 and sym.isalpha() and sym.isascii():
#                     sc += 0.6

#                 if sc <= 0:
#                     continue
#                 symbol_defs[sym].append({
#                     "text": snippet, "para_index": p_idx, "sent_index": s_idx,
#                     "score": round(sc, 3),
#                 })

#     records: list[dict[str, Any]] = []
#     for sym, _ in sym_freq.most_common():
#         defs = sorted(symbol_defs.get(sym, []), key=lambda d: d["score"], reverse=True)[:max_defs_per_symbol]

#         # 태그 분류
#         # records 만들 때 태깅 부분 교체/추가
#         tags: list[str] = []
#         # 집합 기호: 표준만 남김
#         if sym in {"R","N","Z","Q","C"}: 
#             tags.append("set")

#         # 그리스/파라미터
#         if sym in {"theta","phi","varphi","beta","alpha","sigma","lambda","mu","gamma","delta","epsilon","eta","kappa","rho","tau","psi","omega"}:
#             tags.append("param")

#         # 일반 변수
#         if sym in {"x","y","z","s","t"}:
#             tags.append("var")

#         # 연산자(유니코드)
#         if sym in OP_UNI:
#             tags.append("operator")

#         # 인덱스 후보: 소문자만
#         if len(sym) == 1 and sym.isalpha() and sym.islower():
#             tags.append("maybe_index")

#         # 확률 연산자(P) 힌트
#         if sym == "P":
#             texs = [eq_map.get(eid, "") for eid in ids]
#             if any(re.search(r"(\\Pr|\\mathbb\{P\}|(?<![A-Za-z])P\s*\()", t) for t in texs):
#                 tags.append("operator")
#                 tags.append("prob")


#         # eq_ids 정리
#         ids = list(dict.fromkeys(sym_eqs.get(sym, [])))[:5]

#         # 클래스 인덱스 힌트: \sum_{...c...} 또는 (...c...) 패턴
#         if sym == "c":
#             texs = [eq_map.get(eid, "") for eid in ids]
#             if any(re.search(r"\\sum_\{[^}]*c[^}]*\}", t) or re.search(r"\([^\)]*c[^\)]*\)", t) for t in texs):
#                 tags.append("class_index")
#                 if "maybe_index" in tags:
#                     tags.remove("maybe_index")

#         records.append({
#             "symbol": sym,
#             "defs": defs,
#             "freq_eq": int(len(ids)),                 # 등장 eq 개수
#             "first_eq_id": ids[0] if ids else None,
#             "eq_ids": ids,
#             "tags": tags,
#         })
#     return records

# # -------- 프리림 요약 --------
# SECTION_HINTS = re.compile(r"(?i)\b(preliminar(?:y|ies)|notation|assumption|problem\s+setup|objective)\b")

# def summarize_preliminaries(paragraphs: list[str], max_chars: int = 1200) -> str:
#     scored: list[tuple[float, str]] = []
#     for p in paragraphs:
#         txt = _norm_space(p)
#         if not txt: continue
#         base = 0.0
#         if SECTION_HINTS.search(txt): base += 2.0
#         if CUES_RE.search(txt): base += 1.0
#         if "⟨EQ:" in txt: base += 0.4
#         scored.append((base, txt))
#     out, total = [], 0
#     for _, t in sorted(scored, key=lambda x: x[0], reverse=True):
#         if total + len(t) + 2 > max_chars: break
#         out.append(t); total += len(t) + 2
#     return "\n".join(out)

# # -------- Qwen에 전달할 '내용만' 생성 --------
# def build_qwen_math_payload(
#     question: str,
#     equations: list[dict[str, Any]],
#     symbol_table: list[dict[str, Any]],
#     prelim_summary: str | None = None,
#     max_equations: int = 12,
#     max_defs_per_symbol: int = 2,
# ) -> dict[str, Any]:
#     """
#     role 없이 전달할 페이로드 생성:
#     {
#       "question": "...",
#       "prelim_summary": "...",
#       "equations": [{"id","env","tex"}...],
#       "symbol_glossary": [{"symbol","defs":[text..], "freq_eq", "tags"}...],
#       "prompt_text": "최종 한 덩어리 텍스트(선택)"
#     }
#     """
#     # 수식 셀렉션: display 우선
#     disp = [e for e in equations if e.get("display", True)]
#     ranked_eqs = disp[:max_equations]

#     # 기호 사전 압축
#     sym_sorted = sorted(symbol_table, key=lambda r: (-r["freq_eq"], r["symbol"]))
#     gloss = []
#     for rec in sym_sorted[:20]:
#         if "operator" in rec.get("tags", []) or "format" in rec.get("tags", []) or "label" in rec.get("tags", []):
#             continue  # 연산자/포맷/라벨 제외
#         gloss.append({
#             "symbol": rec["symbol"],
#             "defs": [d["text"] for d in rec["defs"][:max_defs_per_symbol]],
#             "freq_eq": rec["freq_eq"],
#             "tags": rec.get("tags", []),
#         })

#     # 한 덩어리 텍스트(원하면 바로 user 메시지로 써도 됨)
#     prelim = f"Preliminaries / Notation:\n{prelim_summary}\n\n" if prelim_summary else ""
#     eq_block = "\n".join([f"({i+1}) $$ {e['tex']} $$" for i, e in enumerate(ranked_eqs)])
#     gloss_block = "\n".join([f"- {g['symbol']}: " + ("; ".join(g["defs"]) if g["defs"] else f"(used {g['freq_eq']}×)") for g in gloss])

#     prompt_text = (
#         f"{prelim}"
#         "Key Equations:\n" + eq_block +
#         "\n\nSymbol Glossary:\n" + gloss_block +
#         "\n\nTask:\n" + question.strip()
#     )

#     return {
#         "question": question.strip(),
#         "prelim_summary": prelim_summary or "",
#         "equations": [{"id": e.get("id"), "env": e.get("env"), "tex": e.get("tex")} for e in ranked_eqs],
#         "symbol_glossary": gloss,
#         "prompt_text": prompt_text,
#     }

# # -------- 엔드-투-엔드 헬퍼 --------
# def build_for_math_llm(
#     body_text: str,
#     equations: list[dict[str, Any]],
#     question: str,
#     prelim_chars: int = 1200,
# ) -> dict[str, Any]:
#     """
#     본문/수식/질문을 받아 페이로드 한 번에 생성.
#     """
#     paragraphs = _split_paragraphs(body_text)
#     symtab = build_symbol_table(paragraphs, equations)
#     prelim = summarize_preliminaries(paragraphs, max_chars=prelim_chars)
#     payload = build_qwen_math_payload(
#         question=question,
#         equations=equations,
#         symbol_table=symtab,
#         prelim_summary=prelim,
#     )
#     return {
#         "symbol_table": symtab,
#         "prelim_summary": prelim,
#         "equations_selected": payload["equations"],
#         "qwen_payload": payload,  # role 없음. caller가 알아서 감쌀 것.
#     }
