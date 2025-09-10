# server/routes/files.py
from fastapi import APIRouter, HTTPException, Query
import logging
from typing import Optional, List, Dict, Any
from services.file_manager import file_manager
from services.environment import env_manager

router = APIRouter(tags=["files"])
logger = logging.getLogger(__name__)

@router.get("/files")
async def get_files(
    file_type: str = Query(default="all", description="파일 타입: all, origin, easy, math"),
    user_id: Optional[str] = Query(default=None, description="사용자 ID (선택사항)")
):
    """파일 목록 조회"""
    try:
        files = file_manager.get_file_list(file_type, user_id)
        return {
            "files": files,
            "count": len(files),
            "file_type": file_type,
            "storage_info": env_manager.get_storage_info(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"파일 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 목록 조회에 실패했습니다: {e}")

@router.get("/files/stats")
async def get_file_stats():
    """파일 통계 조회"""
    try:
        all_files = file_manager.get_file_list("all")
        origin_files = [f for f in all_files if f["file_type"] == "origin"]
        easy_files = [f for f in all_files if f["file_type"] == "easy"]
        math_files = [f for f in all_files if f["file_type"] == "math"]
        
        # 파일 크기 통계
        total_size = sum(f["file_size"] for f in all_files)
        origin_size = sum(f["file_size"] for f in origin_files)
        easy_size = sum(f["file_size"] for f in easy_files)
        math_size = sum(f["file_size"] for f in math_files)
        
        return {
            "total_files": len(all_files),
            "origin_files": len(origin_files),
            "easy_files": len(easy_files),
            "math_files": len(math_files),
            "total_size_bytes": total_size,
            "origin_size_bytes": origin_size,
            "easy_size_bytes": easy_size,
            "math_size_bytes": math_size,
            "storage_info": env_manager.get_storage_info(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"파일 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 통계 조회에 실패했습니다: {e}")

@router.get("/files/{file_id}")
async def get_file_info(file_id: str):
    """특정 파일 정보 조회"""
    try:
        all_files = file_manager.get_file_list("all")
        file_info = next((f for f in all_files if f["filename"] == file_id), None)
        
        if not file_info:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        return {
            "file": file_info,
            "storage_info": env_manager.get_storage_info(),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 정보 조회에 실패했습니다: {e}")

@router.get("/environment/status")
async def get_environment_status():
    """환경 상태 조회"""
    try:
        return {
            "environment": env_manager.get_storage_info(),
            "file_manager": {
                "raw_dir": str(file_manager.raw_dir),
                "outputs_dir": str(file_manager.outputs_dir)
            },
            "status": "success"
        }
    except Exception as e:
        logger.error(f"환경 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"환경 상태 조회에 실패했습니다: {e}")


