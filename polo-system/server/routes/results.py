from fastapi import APIRouter, HTTPException, Query
from starlette.responses import FileResponse
from pathlib import Path
import json
import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(tags=["easy-results"])
logger = logging.getLogger(__name__)

# 데이터 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # polo-system 루트
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# -------- 유틸 --------
SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z가-힣._-]+")

def _sanitize_filename(name: str) -> str:
    name = Path(name).name  # 경로 성분 제거
    name = name.strip()[:200]
    name = SAFE_NAME_RE.sub("_", name)
    return name

def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON 로드 실패 {p}: {e}")
        return None

def _meta_processed_at(meta: Dict[str, Any]) -> Optional[datetime]:
    val = meta.get("processed_at")
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except Exception:
        return None

def _plain_preview(text: str, n: int = 200) -> str:
    if not text:
        return ""
    return (text[:n] + "...") if len(text) > n else text

# -------- 엔드포인트 --------
@router.get("/results")
async def get_all_results(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    q: Optional[str] = Query(None, description="파일명/제목/요약 검색어"),
    since: Optional[str] = Query(None, description="ISO 시각 이후 처리분만"),
    until: Optional[str] = Query(None, description="ISO 시각 이전 처리분만"),
):
    """
    모든 처리 결과 목록(페이지네이션 + 필터)
    - page, per_page
    - q: filename / title / plain_summary에 부분매칭
    - since/until: processed_at 기준(없으면 파일 mtime 기준)
    """
    try:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
        except Exception:
            raise HTTPException(status_code=400, detail="since 파라미터는 ISO8601 형식이어야 합니다.")
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00")) if until else None
        except Exception:
            raise HTTPException(status_code=400, detail="until 파라미터는 ISO8601 형식이어야 합니다.")

        items = []
        for jf in OUTPUTS_DIR.glob("*.json"):
            data = _load_json(jf)
            if not data:
                continue
            metadata = data.get("metadata", {})
            processed_at_dt = _meta_processed_at(metadata) or datetime.fromtimestamp(jf.stat().st_mtime)
            title = data.get("title", "")
            plain_summary = data.get("plain_summary", "")

            # 기간 필터
            if since_dt and processed_at_dt < since_dt:
                continue
            if until_dt and processed_at_dt > until_dt:
                continue

            # 검색어 필터
            if q:
                ql = q.lower()
                if not (
                    (ql in jf.name.lower()) or
                    (ql in (title or "").lower()) or
                    (ql in (plain_summary or "").lower())
                ):
                    continue

            items.append({
                "filename": jf.name,
                "original_filename": metadata.get("original_filename", "Unknown"),
                "processed_at": metadata.get("processed_at", processed_at_dt.isoformat()),
                "file_size": metadata.get("file_size", 0),
                "title": title,
                "plain_summary": _plain_preview(plain_summary, 200),
                "processing_info": data.get("processing_info", {}),
                "json_path": str(jf),
            })

        # 최신순 정렬
        items.sort(key=lambda x: x.get("processed_at", ""), reverse=True)

        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        return {
            "total_count": total,
            "page": page,
            "per_page": per_page,
            "results": page_items,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"결과 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")

@router.get("/results/{filename}")
async def get_result_by_filename(filename: str):
    """특정 결과(JSON) 조회"""
    safe = _sanitize_filename(filename)
    if not safe.endswith(".json"):
        safe += ".json"
    jf = OUTPUTS_DIR / safe
    if not jf.exists():
        raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
    data = _load_json(jf)
    if data is None:
        raise HTTPException(status_code=500, detail="결과 파일을 읽을 수 없습니다.")
    return {"filename": jf.name, "data": data}

@router.get("/results/by-doc/{doc_id}")
async def get_result_by_doc_id(doc_id: str):
    """
    업로드 때 부여한 doc_id(raw PDF 파일명)로 결과 탐색.
    metadata.doc_id == doc_id 인 JSON을 최신순으로 반환(최신 1개).
    """
    safe_doc = _sanitize_filename(doc_id)
    candidates = []
    for jf in OUTPUTS_DIR.glob("*.json"):
        data = _load_json(jf)
        if not data:
            continue
        meta = data.get("metadata", {})
        if meta.get("doc_id") == safe_doc:
            pa = _meta_processed_at(meta) or datetime.fromtimestamp(jf.stat().st_mtime)
            candidates.append((pa, jf.name, data))
    if not candidates:
        raise HTTPException(status_code=404, detail="해당 doc_id의 결과가 없습니다.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, fname, data = candidates[0]
    return {"filename": fname, "data": data}

@router.get("/download/{filename}")
async def download_result(filename: str):
    """결과 JSON 파일 다운로드"""
    safe = _sanitize_filename(filename)
    if not safe.endswith(".json"):
        safe += ".json"
    jf = OUTPUTS_DIR / safe
    if not jf.exists():
        raise HTTPException(status_code=404, detail="결과 파일이 존재하지 않습니다.")
    return FileResponse(str(jf), media_type="application/json", filename=safe)

@router.get("/download/raw/{doc_id}")
async def download_raw_pdf(doc_id: str):
    """원본 PDF 다운로드(doc_id는 RAW 파일명)"""
    safe = _sanitize_filename(doc_id)
    if not safe.endswith(".pdf"):
        # doc_id가 이미 .pdf 포함이면 유지
        candidate_pdf = RAW_DIR / safe
        if not candidate_pdf.exists():
            safe = f"{safe}.pdf"
    pf = RAW_DIR / safe
    if not pf.exists():
        raise HTTPException(status_code=404, detail="원본 PDF를 찾을 수 없습니다.")
    return FileResponse(str(pf), media_type="application/pdf", filename=safe)

@router.get("/results/recent")
async def get_recent_results(limit: int = Query(5, ge=1, le=50)):
    """최근 처리 결과 상위 N개"""
    try:
        json_files = sorted(OUTPUTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        results = []
        for jf in json_files[:limit]:
            data = _load_json(jf)
            if not data:
                continue
            meta = data.get("metadata", {})
            results.append({
                "filename": jf.name,
                "original_filename": meta.get("original_filename", "Unknown"),
                "processed_at": meta.get("processed_at", ""),
                "title": data.get("title", ""),
                "plain_summary": _plain_preview(data.get("plain_summary", ""), 200),
                "processing_info": data.get("processing_info", {}),
            })
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.error(f"최근 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"최근 결과 조회 실패: {str(e)}")

@router.delete("/results/{filename}")
async def delete_result(filename: str):
    """특정 결과 JSON 및 (있다면) 대응 원본 PDF 삭제"""
    safe = _sanitize_filename(filename)
    if not safe.endswith(".json"):
        safe += ".json"
    jf = OUTPUTS_DIR / safe
    if not jf.exists():
        raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")

    # 원본 PDF 추정 파일명 (메타에서 우선 찾고, 없으면 이름 기반 유추)
    deleted = []
    raw_candidate = None
    try:
        data = _load_json(jf) or {}
        meta = data.get("metadata", {})
        did = meta.get("doc_id")
        if did:
            raw_candidate = RAW_DIR / _sanitize_filename(did)
        else:
            # 같은 접두의 타임스탬프 기반 추정
            raw_candidate = RAW_DIR / _sanitize_filename(jf.stem + ".pdf")
    except Exception:
        pass

    # 먼저 JSON 삭제
    try:
        jf.unlink()
        deleted.append(str(jf))
    except Exception as e:
        logger.error(f"JSON 삭제 실패 {jf}: {e}")
        raise HTTPException(status_code=500, detail="JSON 삭제에 실패했습니다.")

    # 그 다음 원본 PDF 삭제(있으면)
    if raw_candidate and raw_candidate.exists():
        try:
            raw_candidate.unlink()
            deleted.append(str(raw_candidate))
            logger.info(f"원본 PDF 삭제: {raw_candidate}")
        except Exception as e:
            logger.warning(f"원본 PDF 삭제 실패 {raw_candidate}: {e}")

    return {"message": f"{safe} 삭제 완료", "deleted_files": deleted}

@router.get("/stats")
async def get_processing_stats():
    """처리 통계"""
    try:
        json_files = list(OUTPUTS_DIR.glob("*.json"))
        pdf_files = list(RAW_DIR.glob("*.pdf"))

        total_processing_time = 0.0
        gpu_usage_count = 0
        for jf in json_files:
            data = _load_json(jf)
            if not data:
                continue
            info = data.get("processing_info", {})
            try:
                total_processing_time += float(info.get("total_time", 0) or 0)
            except Exception:
                pass
            if info.get("gpu_used"):
                gpu_usage_count += 1

        total = len(json_files)
        avg = (total_processing_time / total) if total else 0.0
        rate = (gpu_usage_count / total * 100.0) if total else 0.0

        return {
            "total_processed": total,
            "total_raw_files": len(pdf_files),
            "total_processing_time": round(total_processing_time, 2),
            "average_processing_time": round(avg, 2),
            "gpu_usage_count": gpu_usage_count,
            "gpu_usage_rate": round(rate, 1),
        }
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")
