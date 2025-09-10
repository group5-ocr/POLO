# server/routes/generate.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Union
import logging
import json
import re
from datetime import datetime
from pathlib import Path

from services.llm_client import easy_llm

router = APIRouter(tags=["easy-generate"])
logger = logging.getLogger(__name__)

# 데이터 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=10, description="원문 텍스트")
    filename: Optional[str] = Field(
        None, description="저장시 사용할 베이스 파일명(확장자 제외)"
    )
    # 필요 시 옵션 확장
    # return_full: bool = False

def _sanitize_filename(name: str) -> str:
    # 파일명 안전화: 한글/영문/숫자/._- 만 허용
    name = name.strip()[:200]
    name = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name)
    return name or "doc"

def minimize_easy_json(data: dict) -> dict:
    try:
        result = dict(data)
        for section in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
            sec = (result.get(section) or {})
            if isinstance(sec, dict) and "original" in sec:
                sec.pop("original", None)
                result[section] = sec
        return result
    except Exception as e:
        logger.warning(f"경량화 실패, 원본 저장으로 대체: {e}")
        return data

def _ensure_dict(result: Union[str, dict]) -> dict:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception as e:
            raise ValueError(f"모델이 반환한 문자열을 JSON으로 파싱하지 못했습니다: {e}")
    raise TypeError(f"지원하지 않는 결과 타입: {type(result)}")

@router.post("/generate")
async def generate_from_text(req: GenerateRequest):
    """
    원문 텍스트를 받아 모델에서 easy_json 생성 후 저장
    - 입력: text(필수), filename(선택)
    - 출력: 저장 경로와 경량화된 easy_json
    """
    # 1) 헬스체크
    try:
        healthy = easy_llm.health_check()
    except Exception as e:
        logger.error(f"easy_llm.health_check 예외: {e}")
        raise HTTPException(status_code=503, detail="AI 모델 서비스 연결 실패(health). 프로세스 기동/포트 확인이 필요합니다.")

    if not healthy:
        raise HTTPException(status_code=503, detail="AI 모델 서비스가 사용 불가능합니다. 모델 프로세스를 먼저 기동하세요.")

    # 2) 생성 호출
    try:
        raw = easy_llm.generate(req.text)  # 동기 호출 가정
        data = _ensure_dict(raw)
    except (ValueError, TypeError) as e:
        logger.error(f"모델 결과 파싱 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 결과 파싱 실패: {e}")
    except Exception as e:
        # 입력 길이만 로깅(민감정보 보호)
        logger.exception(f"easy 변환 호출 실패 (len={len(req.text)}): {e}")
        raise HTTPException(status_code=500, detail="AI 모델 처리 중 내부 오류가 발생했습니다.")

    # 3) 메타데이터 보강
    try:
        data.setdefault("metadata", {})
        data["metadata"]["processed_at"] = datetime.now().isoformat()
        if req.filename:
            data["metadata"]["original_filename"] = req.filename
    except Exception as e:
        logger.warning(f"메타데이터 보강 실패: {e}")

    # 4) 저장(파일명 안전화)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = _sanitize_filename(req.filename) if req.filename else f"manual_{timestamp}"

    minimized = minimize_easy_json(data)  # 기본 경량화
    json_file_path = OUTPUTS_DIR / f"{timestamp}_{base_name}.json"

    try:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(minimized, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        raise HTTPException(status_code=500, detail="결과 파일 저장에 실패했습니다.")

    # 5) 응답
    return {
        "status": "success",
        "json_file_path": str(json_file_path),
        "easy_json": minimized,
        # "full_json": data if req.return_full else None,
    }