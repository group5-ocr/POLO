import os
import httpx
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

EASY_MODEL_URL = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
CALLBACK_URL = os.getenv("CALLBACK_URL", "http://localhost:8000/generate/easy-callback")
TIMEOUT = int(os.getenv("EASY_TIMEOUT", "180"))

logger = logging.getLogger("easy_client")
logger.setLevel(logging.INFO)


async def _send_easy_request(text: str) -> Optional[str]:
    """
    Easy 모델의 /simplify API 호출.
    """
    url = f"{EASY_MODEL_URL}/simplify"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, json={"text": text})
            resp.raise_for_status()
            data = resp.json()
            return data.get("simplified_text", "")
    except Exception as e:
        logger.error(f"❌ Easy 모델 호출 실패: {e}")
        return None


async def ingest_jsonl(paper_id: str, jsonl_path: str):
    """
    JSONL 파일을 읽어서 Easy 모델에 병렬 요청하고,
    각 결과를 오케스트레이터 콜백(/generate/easy-callback)으로 전송.
    """
    path = Path(jsonl_path)
    if not path.exists():
        logger.error(f"❌ JSONL 파일 없음: {path}")
        return

    # 1) JSONL 로드
    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    async def process_chunk(index: int, chunk: dict):
        rewritten = await _send_easy_request(chunk["text"])
        if rewritten is None:
            rewritten = ""  # 실패 시 빈 문자열

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                await client.post(
                    CALLBACK_URL,
                    json={
                        "paper_id": str(paper_id),
                        "index": index,
                        "rewritten_text": rewritten
                    },
                )
                logger.info(f"✅ Easy 콜백 전송 완료: paper_id={paper_id}, index={index}")
            except Exception as e:
                logger.error(f"❌ Easy 콜백 전송 실패: {e}")

    # 2) 병렬 처리 (동시성 최적화)
    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(lines)]
    await asyncio.gather(*tasks)

    logger.info(f"🎉 Easy 모델 작업 완료: paper_id={paper_id}, chunks={len(lines)}")