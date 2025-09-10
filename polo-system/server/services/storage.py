# server/services/storage.py
from __future__ import annotations

import os
import json
import logging
import re
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 로컬 저장 경로 (DB 미연결 시 사용): polo-system/data/*
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]           # polo-system 루트
RAW_DIR  = BASE_DIR / "data" / "raw"                     # 원본 PDF
PRE_DIR  = BASE_DIR / "data" / "preprocess"              # 전처리 텍스트
OUT_DIR  = BASE_DIR / "data" / "outputs"                 # 변환 JSON
for _d in (RAW_DIR, PRE_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# DB 모드일 때 JSON 파일 보관 루트(경로만 DB에 저장)
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "storage")).resolve()
(STORAGE_ROOT / "easy").mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DB 연결 감지 및 엔진 준비
#   ⬥ 오직 POSTGRES_* (HOST/PORT/DB/USER/PASSWORD) 만 사용
# ──────────────────────────────────────────────────────────────────────────────
def _compose_url() -> Optional[str]:
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    pwd  = os.getenv("POSTGRES_PASSWORD")
    if not (host and port and db and user and pwd):
        return None
    return f"postgresql+psycopg2://{user}:{quote_plus(pwd)}@{host}:{port}/{db}"

USE_DB: bool = False
_engine = None
_text = None  # lazy import helper

def _try_init_db() -> bool:
    """환경변수(POSTGRES_*)로 DB 접속을 시도하고 성공하면 전역 엔진을 초기화."""
    global USE_DB, _engine, _text
    url = _compose_url()
    if not url:
        logger.info("POSTGRES_* 환경변수 미설정 → 로컬 모드")
        USE_DB = False
        return False
    try:
        from sqlalchemy import create_engine, text
        _engine = create_engine(url, pool_pre_ping=True, future=True)
        _text = text
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ DB 연결 확인 → DB 모드")
        USE_DB = True
        return True
    except Exception as e:
        logger.warning(f"DB 연결 실패 → 로컬 모드 전환: {e}")
        USE_DB = False
        _engine = None
        _text = None
        return False

# 초기 감지
_try_init_db()

# ──────────────────────────────────────────────────────────────────────────────
# DB 유틸 (ERD: users / origin_file / easy_file)
# ──────────────────────────────────────────────────────────────────────────────
_ORIGIN_FN_IS_INT: Optional[bool] = None

def _origin_filename_is_int() -> bool:
    """origin_file.filename 컬럼이 정수형인지 감지 (캐시)."""
    global _ORIGIN_FN_IS_INT
    if _ORIGIN_FN_IS_INT is not None:
        return _ORIGIN_FN_IS_INT
    if not USE_DB:
        _ORIGIN_FN_IS_INT = False
        return False
    q = _text("""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='origin_file'
          AND column_name='filename'
        LIMIT 1
    """)
    with _engine.begin() as conn:
        row = conn.execute(q).first()
    _ORIGIN_FN_IS_INT = bool(row and str(row[0]).lower() in {"integer","int","bigint","smallint","numeric"})
    return _ORIGIN_FN_IS_INT

def _insert_origin_file(*, user_id: int, filename_str: str) -> int:
    """origin_file INSERT 후 origin_id 반환 (filename이 INT면 CRC32로 안전 저장)."""
    if not USE_DB:
        raise RuntimeError("DB 미사용 모드입니다.")
    val: Any = filename_str
    if _origin_filename_is_int():
        val = int(zlib.crc32(filename_str.encode("utf-8")) & 0x7FFFFFFF)
    q = _text("""
        INSERT INTO public.origin_file (user_id, filename)
        VALUES (:uid, :fname)
        RETURNING origin_id
    """)
    with _engine.begin() as conn:
        row = conn.execute(q, {"uid": user_id, "fname": val}).first()
    return int(row[0])

def _insert_easy_file(*, origin_id: int, filename: Optional[str], file_addr: Optional[str]) -> int:
    if not USE_DB:
        raise RuntimeError("DB 미사용 모드입니다.")
    q = _text("""
        INSERT INTO public.easy_file (origin_id, filename, file_addr)
        VALUES (:oid, :fname, :faddr)
        RETURNING easy_id
    """)
    with _engine.begin() as conn:
        row = conn.execute(q, {"oid": origin_id, "fname": filename, "faddr": file_addr}).first()
    return int(row[0])

def _recent_easy(limit: int = 10) -> List[Mapping]:
    if not USE_DB:
        return []
    q = _text("""
        SELECT e.easy_id,
               e.origin_id,
               e.filename      AS easy_filename,
               e.file_addr     AS easy_path,
               o.filename      AS origin_filename,
               o.user_id
        FROM public.easy_file e
        JOIN public.origin_file o ON o.origin_id = e.origin_id
        ORDER BY e.easy_id DESC
        LIMIT :limit
    """)
    with _engine.begin() as conn:
        rows = conn.execute(q, {"limit": limit}).mappings().all()
    return rows

def _get_easy(easy_id: int) -> Optional[Mapping]:
    if not USE_DB:
        return None
    q = _text("""
        SELECT e.easy_id, e.origin_id, e.filename AS easy_filename, e.file_addr AS easy_path,
               o.filename AS origin_filename, o.user_id
        FROM public.easy_file e
        JOIN public.origin_file o ON o.origin_id = e.origin_id
        WHERE e.easy_id = :eid
        LIMIT 1
    """)
    with _engine.begin() as conn:
        row = conn.execute(q, {"eid": easy_id}).mappings().first()
    return row

# ──────────────────────────────────────────────────────────────────────────────
# 공통 도구
# ──────────────────────────────────────────────────────────────────────────────
def _minimize_easy_json(d: dict) -> dict:
    try:
        r = dict(d)
        for k in ["abstract","introduction","methods","results","discussion","conclusion"]:
            sec = (r.get(k) or {})
            if isinstance(sec, dict) and "original" in sec:
                sec.pop("original", None)
                r[k] = sec
        return r
    except Exception:
        return d

def _slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s or "file"

# ──────────────────────────────────────────────────────────────────────────────
# 공통 저장 진입점
# ──────────────────────────────────────────────────────────────────────────────
def save_conversion(
    *,
    original_filename: str,
    pdf_bytes: bytes,
    extracted_text: str,
    easy_json: Dict[str, Any],
    processing_info: Optional[Dict[str, Any]] = None,
    user_id: int = 0,
) -> Dict[str, Any]:
    """
    변환 결과 저장(메타/JSON). DB 연결되면 DB+파일경로, 아니면 data/* 로컬 저장.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = _slug(Path(original_filename).stem)

    # 공통 메타 부착 + 경량화
    easy_json.setdefault("metadata", {})
    easy_json["metadata"].update({
        "original_filename": original_filename,
        "processed_at": datetime.now().isoformat(),
        "file_size": len(pdf_bytes),
        "extracted_text_length": len(extracted_text),
    })
    if processing_info:
        easy_json["processing_info"] = processing_info
    minimized = _minimize_easy_json(easy_json)

    # ── DB 모드 ───────────────────────────────────────────────────────────────
    if USE_DB:
        origin_id = _insert_origin_file(user_id=user_id, filename_str=base_name)

        # JSON 파일 외부 저장 (원하면 DB JSONB로도 저장 가능)
        easy_dir = STORAGE_ROOT / "easy"
        easy_dir.mkdir(parents=True, exist_ok=True)
        json_name = f"easy_{origin_id}_{ts}.json"
        json_path = easy_dir / json_name
        json_path.write_text(json.dumps(minimized, ensure_ascii=False, indent=2), encoding="utf-8")

        easy_id = _insert_easy_file(origin_id=origin_id, filename=json_name, file_addr=str(json_path))

        # 전처리 텍스트 보관(선택)
        pre_dir = STORAGE_ROOT / "preprocess"
        pre_dir.mkdir(parents=True, exist_ok=True)
        (pre_dir / f"{ts}_{base_name}.txt").write_text(extracted_text, encoding="utf-8")

        return {
            "mode": "db",
            "status": "success",
            "origin_id": origin_id,
            "easy_id": easy_id,
            "easy_json_path": str(json_path),
        }

    # ── 로컬 모드 ─────────────────────────────────────────────────────────────
    raw_path = RAW_DIR / f"{ts}_{base_name}.pdf"
    raw_path.write_bytes(pdf_bytes)

    pre_path = PRE_DIR / f"{ts}_{base_name}.txt"
    pre_path.write_text(extracted_text, encoding="utf-8")

    out_path = OUT_DIR / f"{ts}_{base_name}.json"
    out_path.write_text(json.dumps(minimized, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mode": "local",
        "status": "success",
        "raw_file_path": str(raw_path),
        "preprocess_path": str(pre_path),
        "json_file_path": str(out_path),
        "filename": out_path.name,  # 상세/다운로드 키로 사용
    }

# ──────────────────────────────────────────────────────────────────────────────
# 목록/상세/다운로드
# ──────────────────────────────────────────────────────────────────────────────
def recent_results(limit: int = 10) -> List[Dict[str, Any]]:
    if USE_DB:
        rows = _recent_easy(limit)
        out: List[Dict[str, Any]] = []
        for r in rows:
            title = ""
            plain_summary = ""
            processing_info = {}
            p = r.get("easy_path")
            if p and Path(p).exists():
                try:
                    j = json.loads(Path(p).read_text(encoding="utf-8"))
                    title = j.get("title","")
                    ps = j.get("plain_summary","") or ""
                    plain_summary = (ps[:200] + "...") if len(ps) > 200 else ps
                    processing_info = j.get("processing_info", {}) or {}
                except Exception as e:
                    logger.warning(f"JSON 읽기 실패({p}): {e}")
            out.append({
                "mode": "db",
                "easy_id": int(r["easy_id"]),
                "origin_id": int(r["origin_id"]),
                "filename": str(r.get("easy_filename") or f"easy_{r['easy_id']}.json"),
                "original_filename": str(r.get("origin_filename") or ""),
                "processed_at": processing_info.get("processed_at", ""),
                "title": title,
                "plain_summary": plain_summary,
                "processing_info": processing_info,
            })
        return out

    # 로컬
    items: List[Dict[str,Any]] = []
    files = sorted(OUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for p in files:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        md = j.get("metadata", {}) or {}
        items.append({
            "mode": "local",
            "filename": p.name,
            "original_filename": md.get("original_filename","Unknown"),
            "processed_at": md.get("processed_at",""),
            "title": j.get("title",""),
            "plain_summary": j.get("plain_summary",""),
            "processing_info": j.get("processing_info",{}) or {},
            "json_path": str(p),
        })
    return items

def get_result(key: str) -> Dict[str, Any]:
    if USE_DB and key.isdigit():
        r = _get_easy(int(key))
        if not r:
            raise FileNotFoundError("DB에서 결과를 찾을 수 없습니다.")
        path = r.get("easy_path")
        data = {}
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {"mode": "db", "easy_id": int(r["easy_id"]), "origin_id": int(r["origin_id"]), "data": data}

    # 로컬
    p = OUT_DIR / key
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    return {"mode": "local", "filename": key, "data": data}

def read_json_for_download(key: str) -> str:
    if USE_DB and key.isdigit():
        r = _get_easy(int(key))
        if not r:
            raise FileNotFoundError("DB에서 결과를 찾을 수 없습니다.")
        path = r.get("easy_path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(path or "경로없음")
        return Path(path).read_text(encoding="utf-8")

    # 로컬
    p = OUT_DIR / key
    if not p.exists():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8")
