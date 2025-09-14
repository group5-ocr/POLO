import os
import json
import httpx
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("preprocess_client")
logger.setLevel(logging.INFO)

PREPROCESS_URL = os.getenv("PREPROCESS_URL", "http://localhost:5002")
MATH_MODEL_URL = os.getenv("MATH_MODEL_URL", "http://localhost:5004")
EASY_MODEL_URL = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
CALLBACK_URL   = os.getenv("CALLBACK_URL", "http://localhost:8000").rstrip("/")

TIMEOUT = int(os.getenv("PREPROCESS_TIMEOUT", "300"))
MATH_TIMEOUT = int(os.getenv("MATH_TIMEOUT", "300"))
EASY_TIMEOUT = int(os.getenv("EASY_TIMEOUT", "600"))

# ─────────────────────────────────────────────────────────────
# 헬퍼: 출력 폴더에서 산출물 자동 탐색 (관습 기반)
# ─────────────────────────────────────────────────────────────
def _find_chunks_jsonl(out_dir: Path) -> Optional[Path]:
    # 관례: *.jsonl 가 텍스트 청크 (easy용)
    hits = list(out_dir.rglob("*.jsonl"))
    return hits[0] if hits else None

def _find_tex_roots(source_dir: Path) -> List[Path]:
    # 관례: source_dir 내 *.tex 들이 math용
    return [p for p in source_dir.rglob("*.tex") if p.is_file()]

# ─────────────────────────────────────────────────────────────
# 메인 오케스트레이션
# ─────────────────────────────────────────────────────────────
async def run(paper_id: str, source_dir: str, callback: str):
    """
    1) 전처리 호출
    2) 전처리 산출물 탐색 (jsonl, tex)
    3) 수학모델에 LaTeX 병합 요청
    4) 이지모델에 JSONL 배치 요청
    5) 콜백으로 종합 결과 통지
    """
    source_dir_p = Path(source_dir).resolve()
    out_dir_p = Path(f"data/outputs/{paper_id}").resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    preprocess_result: Dict[str, Any] = {}
    math_result: Dict[str, Any] = {}
    easy_result: Dict[str, Any] = {}

    # 1) Preprocess
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{PREPROCESS_URL}/preprocess",
                json={
                    "input_path": str(source_dir_p),
                    "output_dir": str(out_dir_p),
                    "config_path": "configs/default.yaml",
                },
            )
            resp.raise_for_status()
            preprocess_result = resp.json()
            logger.info(f"✅ 전처리 완료: paper_id={paper_id}, out_dir={preprocess_result.get('out_dir', str(out_dir_p))}")
    except Exception as e:
        logger.error(f"❌ 전처리 요청 실패: {e}")
        # 실패도 콜백에 보고
        await _post_callback(callback, {
            "paper_id": paper_id,
            "status": "preprocess_failed",
            "error": str(e),
        })
        raise

    # 2) 산출물 탐색
    #   - easy용: jsonl (텍스트 청크)
    #   - math용: source_dir 에서 tex들 (병합 대상)
    chunks_jsonl = None
    if "chunks_path" in preprocess_result:
        chunks_jsonl = Path(preprocess_result["chunks_path"]).resolve()
    else:
        chunks_jsonl = _find_chunks_jsonl(out_dir_p)

    tex_files = _find_tex_roots(source_dir_p)

    # 3) Math: LaTeX 병합 (엔드포인트는 구현에 맞춰 조정)
    #    가정: POST /merge  body: {"paper_id": ..., "tex_paths": [...], "output_dir": "..."}
    try:
        if tex_files:
            payload = {
                "paper_id": paper_id,
                "tex_paths": [str(p) for p in tex_files],
                "output_dir": str(out_dir_p / "math_merged"),
            }
            async with httpx.AsyncClient(timeout=MATH_TIMEOUT) as client:
                r = await client.post(f"{MATH_MODEL_URL}/merge", json=payload)
                r.raise_for_status()
                math_result = r.json()
                logger.info(f"✅ Math 병합 완료: {math_result}")
        else:
            logger.warning("⚠️ 병합할 tex 파일이 없어 math 단계는 건너뜀.")
            math_result = {"skipped": True, "reason": "no_tex_found"}
    except Exception as e:
        logger.error(f"❌ Math 병합 실패: {e}")
        math_result = {"ok": False, "error": str(e)}

    # 4) Easy: JSONL 배치 (엔드포인트는 구현에 맞춰 조정)
    #    가정: POST /batch  body: {"paper_id": ..., "chunks_jsonl": "...", "output_dir": "..."}
    try:
        if chunks_jsonl and chunks_jsonl.exists():
            payload = {
                "paper_id": paper_id,
                "chunks_jsonl": str(chunks_jsonl),
                "output_dir": str(out_dir_p / "easy_outputs"),
            }
            async with httpx.AsyncClient(timeout=EASY_TIMEOUT) as client:
                r = await client.post(f"{EASY_MODEL_URL}/batch", json=payload)
                r.raise_for_status()
                easy_result = r.json()
                logger.info(f"✅ Easy 배치 완료: {easy_result}")
        else:
            logger.warning("⚠️ JSONL 산출물을 찾지 못해 easy 단계는 건너뜀.")
            easy_result = {"skipped": True, "reason": "no_jsonl_found"}
    except Exception as e:
        logger.error(f"❌ Easy 배치 실패: {e}")
        easy_result = {"ok": False, "error": str(e)}

    # 5) 콜백 통지
    payload = {
        "paper_id": paper_id,
        "status": "done",
        "preprocess": preprocess_result,
        "math": math_result,
        "easy": easy_result,
        "transport_path": str(out_dir_p),  # 네가 쓰는 필드 유지
    }
    await _post_callback(callback, payload)
    return payload


async def _post_callback(callback: str, payload: Dict[str, Any]):
    """오케스트레이터 콜백으로 결과/상태 보고"""
    # callback이 절대경로가 아니면 CALLBACK_URL prepend
    if callback.startswith("http://") or callback.startswith("https://"):
        url = callback
    else:
        url = f"{CALLBACK_URL.rstrip('/')}/{callback.lstrip('/')}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info(f"🔔 콜백 전송 성공 → {url}")
    except Exception as e:
        logger.error(f"❌ 콜백 전송 실패({url}): {e}")
