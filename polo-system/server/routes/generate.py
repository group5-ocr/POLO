from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
import json
from datetime import datetime
from pathlib import Path
from services.llm_client import easy_llm

router = APIRouter()
logger = logging.getLogger(__name__)

# 데이터 디렉토리 경로
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=10, description="원문 텍스트")
    filename: Optional[str] = Field(None, description="저장시 사용할 베이스 파일명(확장자 제외)")


def minimize_easy_json(data: dict) -> dict:
    try:
        result = dict(data)
        for section in ["abstract","introduction","methods","results","discussion","conclusion"]:
            sec = (result.get(section) or {})
            if isinstance(sec, dict) and "original" in sec:
                sec.pop("original", None)
                result[section] = sec
        return result
    except Exception as e:
        logger.warning(f"경량화 실패, 원본을 저장합니다: {e}")
        return data


@router.post("/generate")
async def generate_from_text(req: GenerateRequest):
    """원문 텍스트를 받아 모델에서 easy_json 생성 후 저장"""
    if not easy_llm.health_check():
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다.")

    try:
        # 모델 호출
        result = easy_llm.generate(req.text)
        if result is None:
            raise HTTPException(status_code=500, detail="AI 모델 처리 중 오류가 발생했습니다.")

        # 메타데이터 보강
        if isinstance(result, dict):
            result.setdefault("metadata", {})
            result["metadata"]["processed_at"] = datetime.now().isoformat()
            if req.filename:
                result["metadata"]["original_filename"] = req.filename

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = (req.filename or f"manual_{timestamp}")
        minimized = minimize_easy_json(result)
        json_file_path = OUTPUTS_DIR / f"{timestamp}_{base_name}.json"
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(minimized, f, ensure_ascii=False, indent=2)

        return {
            "json_file_path": str(json_file_path),
            "easy_json": minimized,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"텍스트 변환 실패: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 변환 실패: {str(e)}")


