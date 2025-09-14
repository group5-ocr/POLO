from __future__ import annotations

import os
import httpx
import logging
from typing import Dict, Any

EASY_MODEL_URL = os.getenv("EASY_MODEL_URL", "http://localhost:5003")
TIMEOUT = int(os.getenv("EASY_TIMEOUT", "600"))

logger = logging.getLogger("easy_client")
logger.setLevel(logging.INFO)

async def run_batch(paper_id: str, jsonl_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Easy /batch 호출: 재서술 → Viz 시각화(/viz) → PNG 저장을 한 번에 수행.
    """
    url = f"{EASY_MODEL_URL.rstrip('/')}/batch"
    payload = {
        "paper_id": str(paper_id),
        "chunks_jsonl": jsonl_path,
        "output_dir": output_dir,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        res = r.json()
        logger.info(
            "✅ Easy batch 완료: paper_id=%s success=%s failed=%s",
            paper_id, res.get("success"), res.get("failed")
        )
        return res
