# server/services/db.py
from __future__ import annotations
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, RowMapping

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("환경변수 DATABASE_URL 이(가) 없습니다.")

# 동기 엔진 (FastAPI에서 충분히 안전)
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

# --- 테이블 레이아웃 가정 ---
# public.origin_file(
#   id bigserial PK,
#   doc_id uuid UNIQUE,
#   original_filename text,
#   file_size bigint,
#   pdf_bytes bytea,
#   extracted_text_length int,
#   extracted_text_preview text,
#   created_at timestamptz default now()
# )
# public.easy_file(
#   id bigserial PK,
#   doc_id uuid REFERENCES origin_file(doc_id) ON DELETE CASCADE,
#   easy_json jsonb,
#   processed_at timestamptz default now(),
#   processing_info jsonb
# )
#
# ※ 테이블이 이미 있다면 컬럼명만 맞춰주면 됩니다.

def save_origin_pdf(
    *,
    original_filename: str,
    file_size: int,
    pdf_bytes: bytes,
    extracted_text_length: int,
    extracted_text_preview: str,
    doc_id: Optional[str] = None,
) -> str:
    """원본 PDF 저장 후 doc_id 반환"""
    _doc_id = doc_id or str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO public.origin_file
                (doc_id, original_filename, file_size, pdf_bytes,
                 extracted_text_length, extracted_text_preview, created_at)
                VALUES (:doc_id, :original_filename, :file_size, :pdf_bytes,
                        :etl, :preview, now())
            """),
            {
                "doc_id": _doc_id,
                "original_filename": original_filename,
                "file_size": file_size,
                "pdf_bytes": pdf_bytes,
                "etl": extracted_text_length,
                "preview": extracted_text_preview,
            },
        )
    return _doc_id

def save_easy_json(
    *,
    doc_id: str,
    easy_json: Dict[str, Any],
    processing_info: Optional[Dict[str, Any]] = None,
) -> int:
    """변환 JSON 저장 후 easy_file.id 반환"""
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO public.easy_file (doc_id, easy_json, processed_at, processing_info)
                VALUES (:doc_id, cast(:easy_json as jsonb), now(), cast(:proc as jsonb))
                RETURNING id
            """),
            {
                "doc_id": doc_id,
                "easy_json": easy_json,
                "proc": processing_info or {},
            },
        ).first()
    return int(row[0])  # easy_id

def get_recent_results(limit: int = 5) -> List[RowMapping]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                  e.id        AS easy_id,
                  e.doc_id    AS doc_id,
                  e.processed_at,
                  e.processing_info,
                  o.original_filename,
                  (e.easy_json->>'title')          AS title,
                  (e.easy_json->>'plain_summary')  AS plain_summary
                FROM public.easy_file e
                LEFT JOIN public.origin_file o USING (doc_id)
                ORDER BY e.processed_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).mappings().all()
    return rows

def get_easy_by_id(easy_id: int) -> Optional[RowMapping]:
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT e.id AS easy_id, e.doc_id, e.easy_json, e.processing_info,
                       e.processed_at, o.original_filename, o.file_size
                FROM public.easy_file e
                LEFT JOIN public.origin_file o USING (doc_id)
                WHERE e.id = :eid
            """),
            {"eid": easy_id},
        ).mappings().first()
    return row

def get_json_for_download(easy_id: int) -> Optional[str]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT easy_json::text AS j FROM public.easy_file WHERE id=:eid"),
            {"eid": easy_id},
        ).first()
    return row[0] if row else None

def get_raw_pdf(doc_id: str) -> Optional[bytes]:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT pdf_bytes FROM public.origin_file WHERE doc_id=:doc_id"),
            {"doc_id": doc_id},
        ).first()
    return bytes(row[0]) if row else None