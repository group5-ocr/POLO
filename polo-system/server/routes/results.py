from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import json
import logging
from typing import List, Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# 데이터 디렉토리 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent  # polo-system 루트
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR = BASE_DIR / "data" / "raw"

@router.get("/results")
async def get_all_results():
    """모든 처리 결과 목록 조회"""
    try:
        results = []
        
        # JSON 파일들 스캔
        for json_file in OUTPUTS_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 메타데이터에서 정보 추출
                metadata = data.get("metadata", {})
                
                results.append({
                    "filename": json_file.name,
                    "original_filename": metadata.get("original_filename", "Unknown"),
                    "processed_at": metadata.get("processed_at", ""),
                    "file_size": metadata.get("file_size", 0),
                    "title": data.get("title", ""),
                    "plain_summary": data.get("plain_summary", "")[:200] + "..." if len(data.get("plain_summary", "")) > 200 else data.get("plain_summary", ""),
                    "processing_info": data.get("processing_info", {}),
                    "json_path": str(json_file)
                })
            except Exception as e:
                logger.error(f"결과 파일 읽기 실패 {json_file}: {e}")
                continue
        
        # 처리 시간 순으로 정렬 (최신순)
        results.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
        
        return {
            "total_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"결과 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")

@router.get("/results/{filename}")
async def get_result_by_filename(filename: str):
    """특정 파일의 처리 결과 조회"""
    try:
        json_file = OUTPUTS_DIR / filename
        
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            "filename": filename,
            "data": data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"결과 파일 조회 실패 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")

@router.get("/results/recent")
async def get_recent_results(limit: int = Query(5, ge=1, le=50)):
    """최근 처리 결과 조회"""
    try:
        results = []
        
        # JSON 파일들을 수정 시간 순으로 정렬
        json_files = sorted(OUTPUTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for json_file in json_files[:limit]:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                metadata = data.get("metadata", {})
                
                results.append({
                    "filename": json_file.name,
                    "original_filename": metadata.get("original_filename", "Unknown"),
                    "processed_at": metadata.get("processed_at", ""),
                    "title": data.get("title", ""),
                    "plain_summary": data.get("plain_summary", "")[:200] + "..." if len(data.get("plain_summary", "")) > 200 else data.get("plain_summary", ""),
                    "processing_info": data.get("processing_info", {})
                })
            except Exception as e:
                logger.error(f"결과 파일 읽기 실패 {json_file}: {e}")
                continue
        
        return {
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"최근 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"최근 결과 조회 실패: {str(e)}")

@router.delete("/results/{filename}")
async def delete_result(filename: str):
    """특정 결과 파일 삭제"""
    try:
        json_file = OUTPUTS_DIR / filename
        
        if not json_file.exists():
            raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다.")
        
        # JSON 파일 삭제
        json_file.unlink()
        
        # 해당하는 원본 PDF 파일도 삭제 (선택사항)
        base_name = filename.replace(".json", ".pdf")
        pdf_file = RAW_DIR / base_name
        if pdf_file.exists():
            pdf_file.unlink()
            logger.info(f"원본 PDF도 삭제: {pdf_file}")
        
        return {
            "message": f"결과 파일 {filename}이 삭제되었습니다.",
            "deleted_files": [str(json_file)] + ([str(pdf_file)] if pdf_file.exists() else [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"결과 파일 삭제 실패 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {str(e)}")

@router.get("/stats")
async def get_processing_stats():
    """처리 통계 조회"""
    try:
        json_files = list(OUTPUTS_DIR.glob("*.json"))
        pdf_files = list(RAW_DIR.glob("*.pdf"))
        
        total_processing_time = 0
        gpu_usage_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                processing_info = data.get("processing_info", {})
                if processing_info.get("total_time"):
                    total_processing_time += processing_info["total_time"]
                if processing_info.get("gpu_used"):
                    gpu_usage_count += 1
            except:
                continue
        
        return {
            "total_processed": len(json_files),
            "total_raw_files": len(pdf_files),
            "total_processing_time": round(total_processing_time, 2),
            "average_processing_time": round(total_processing_time / len(json_files), 2) if json_files else 0,
            "gpu_usage_count": gpu_usage_count,
            "gpu_usage_rate": round(gpu_usage_count / len(json_files) * 100, 1) if json_files else 0
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")
