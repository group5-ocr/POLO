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
async def run_async(paper_id: str, source_dir: str, callback: str):
    """비동기 전처리 실행"""
    return await run(paper_id, source_dir, callback)

async def run(paper_id: str, source_dir: str, callback: str):
    """
    1) 전처리 호출
    2) 전처리 산출물 탐색 (jsonl, tex)
    3) 수학모델에 LaTeX 병합 요청
    4) 이지모델에 JSONL 배치 요청
    5) 콜백으로 종합 결과 통지
    """
    source_dir_p = Path(source_dir).resolve()
    out_dir_p = Path(f"server/data/outputs/{paper_id}").resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    preprocess_result: Dict[str, Any] = {}
    math_result: Dict[str, Any] = {}
    easy_result: Dict[str, Any] = {}

    # 1) Preprocess
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{PREPROCESS_URL}/process",
                json={
                    "paper_id": paper_id,
                    "source_dir": str(source_dir_p),
                    "callback": callback,
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

    # 2) 전처리 서비스에서 math/easy 모델 호출을 이미 처리했으므로
    #    여기서는 콜백만 호출
    logger.info(f"✅ 전처리 완료: paper_id={paper_id}")
    
    # 3) 콜백 통지
    payload = {
        "paper_id": paper_id,
        "status": "completed",
        "preprocess": preprocess_result,
        "transport_path": str(out_dir_p),
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
