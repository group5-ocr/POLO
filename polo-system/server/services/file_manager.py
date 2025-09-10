# server/services/file_manager.py
import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from .environment import env_manager
from .db import get_postgres_client

logger = logging.getLogger(__name__)

class FileManager:
    """파일 관리 통합 클래스 (DB + 로컬 파일 시스템)"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = self.base_dir / "data" / "raw"
        self.outputs_dir = self.base_dir / "data" / "outputs"
        
        # 디렉토리 생성
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def save_origin_file(self, filename: str, content: bytes, user_id: Optional[str] = None) -> Dict[str, Any]:
        """원본 파일 저장 (DB + 로컬 파일 시스템)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = self._sanitize_filename(filename)
        raw_filename = f"{timestamp}_{safe_filename}.pdf"
        raw_file_path = self.raw_dir / raw_filename
        
        # 1. 로컬 파일 시스템에 저장
        try:
            with open(raw_file_path, "wb") as f:
                f.write(content)
            logger.info(f"원본 PDF 저장: {raw_file_path}")
        except Exception as e:
            logger.error(f"원본 저장 실패: {e}")
            raise RuntimeError(f"원본 파일 저장에 실패했습니다: {e}")
        
        # 2. DB에 메타데이터 저장 (학원 네트워크인 경우)
        if env_manager.should_use_database():
            try:
                self._save_origin_file_to_db(
                    filename=raw_filename,
                    original_filename=filename,
                    file_path=str(raw_file_path),
                    file_size=len(content),
                    user_id=user_id
                )
                logger.info(f"원본 파일 메타데이터를 DB에 저장: {raw_filename}")
            except Exception as e:
                logger.warning(f"DB 저장 실패, 로컬 파일만 저장: {e}")
        
        return {
            "filename": raw_filename,
            "original_filename": filename,
            "file_path": str(raw_file_path),
            "file_size": len(content),
            "saved_to_db": env_manager.should_use_database()
        }
    
    def save_easy_file(self, origin_filename: str, easy_json: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Easy 변환 결과 저장 (DB + 로컬 파일 시스템)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = self._sanitize_filename(origin_filename)
        json_filename = f"{timestamp}_{safe_base}.json"
        json_file_path = self.outputs_dir / json_filename
        
        # 1. 로컬 파일 시스템에 저장
        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(easy_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Easy JSON 저장: {json_file_path}")
        except Exception as e:
            logger.error(f"Easy JSON 저장 실패: {e}")
            raise RuntimeError(f"Easy JSON 저장에 실패했습니다: {e}")
        
        # 2. DB에 메타데이터 저장 (학원 네트워크인 경우)
        if env_manager.should_use_database():
            try:
                self._save_easy_file_to_db(
                    filename=json_filename,
                    origin_filename=origin_filename,
                    file_path=str(json_file_path),
                    file_size=os.path.getsize(json_file_path),
                    user_id=user_id
                )
                logger.info(f"Easy 파일 메타데이터를 DB에 저장: {json_filename}")
            except Exception as e:
                logger.warning(f"DB 저장 실패, 로컬 파일만 저장: {e}")
        
        return {
            "filename": json_filename,
            "origin_filename": origin_filename,
            "file_path": str(json_file_path),
            "file_size": os.path.getsize(json_file_path),
            "saved_to_db": env_manager.should_use_database()
        }
    
    def save_math_file(self, origin_filename: str, math_json: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Math 변환 결과 저장 (DB + 로컬 파일 시스템)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = self._sanitize_filename(origin_filename)
        json_filename = f"{timestamp}_{safe_base}_math.json"
        json_file_path = self.outputs_dir / json_filename
        
        # 1. 로컬 파일 시스템에 저장
        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(math_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Math JSON 저장: {json_file_path}")
        except Exception as e:
            logger.error(f"Math JSON 저장 실패: {e}")
            raise RuntimeError(f"Math JSON 저장에 실패했습니다: {e}")
        
        # 2. DB에 메타데이터 저장 (학원 네트워크인 경우)
        if env_manager.should_use_database():
            try:
                self._save_math_file_to_db(
                    filename=json_filename,
                    origin_filename=origin_filename,
                    file_path=str(json_file_path),
                    file_size=os.path.getsize(json_file_path),
                    user_id=user_id
                )
                logger.info(f"Math 파일 메타데이터를 DB에 저장: {json_filename}")
            except Exception as e:
                logger.warning(f"DB 저장 실패, 로컬 파일만 저장: {e}")
        
        return {
            "filename": json_filename,
            "origin_filename": origin_filename,
            "file_path": str(json_file_path),
            "file_size": os.path.getsize(json_file_path),
            "saved_to_db": env_manager.should_use_database()
        }
    
    def get_file_list(self, file_type: str = "all", user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """파일 목록 조회 (DB 우선, 없으면 로컬 파일 시스템)"""
        if env_manager.should_use_database():
            try:
                return self._get_file_list_from_db(file_type, user_id)
            except Exception as e:
                logger.warning(f"DB에서 파일 목록 조회 실패, 로컬 파일 시스템 사용: {e}")
        
        return self._get_file_list_from_local(file_type)
    
    def _save_origin_file_to_db(self, filename: str, original_filename: str, file_path: str, file_size: int, user_id: Optional[str] = None):
        """원본 파일 메타데이터를 DB에 저장"""
        query = """
        INSERT INTO origin_file (filename, original_filename, file_path, file_size, user_id, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (filename, original_filename, file_path, file_size, user_id, datetime.now())
        
        with get_postgres_client() as client:
            client.execute_query(query, params)
    
    def _save_easy_file_to_db(self, filename: str, origin_filename: str, file_path: str, file_size: int, user_id: Optional[str] = None):
        """Easy 파일 메타데이터를 DB에 저장"""
        query = """
        INSERT INTO easy_file (filename, origin_filename, file_path, file_size, user_id, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (filename, origin_filename, file_path, file_size, user_id, datetime.now())
        
        with get_postgres_client() as client:
            client.execute_query(query, params)
    
    def _save_math_file_to_db(self, filename: str, origin_filename: str, file_path: str, file_size: int, user_id: Optional[str] = None):
        """Math 파일 메타데이터를 DB에 저장"""
        query = """
        INSERT INTO math_file (filename, origin_filename, file_path, file_size, user_id, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (filename, origin_filename, file_path, file_size, user_id, datetime.now())
        
        with get_postgres_client() as client:
            client.execute_query(query, params)
    
    def _get_file_list_from_db(self, file_type: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """DB에서 파일 목록 조회"""
        if file_type == "origin":
            query = "SELECT * FROM origin_file ORDER BY created_at DESC"
        elif file_type == "easy":
            query = "SELECT * FROM easy_file ORDER BY created_at DESC"
        elif file_type == "math":
            query = "SELECT * FROM math_file ORDER BY created_at DESC"
        else:
            # 모든 파일 타입 조회
            query = """
            SELECT 'origin' as file_type, filename, original_filename, file_path, file_size, user_id, created_at FROM origin_file
            UNION ALL
            SELECT 'easy' as file_type, filename, origin_filename as original_filename, file_path, file_size, user_id, created_at FROM easy_file
            UNION ALL
            SELECT 'math' as file_type, filename, origin_filename as original_filename, file_path, file_size, user_id, created_at FROM math_file
            ORDER BY created_at DESC
            """
        
        with get_postgres_client() as client:
            return client.execute_query(query)
    
    def _get_file_list_from_local(self, file_type: str) -> List[Dict[str, Any]]:
        """로컬 파일 시스템에서 파일 목록 조회"""
        files = []
        
        if file_type in ["origin", "all"]:
            for file_path in self.raw_dir.glob("*.pdf"):
                files.append({
                    "file_type": "origin",
                    "filename": file_path.name,
                    "original_filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "user_id": None,
                    "created_at": datetime.fromtimestamp(file_path.stat().st_mtime)
                })
        
        if file_type in ["easy", "all"]:
            for file_path in self.outputs_dir.glob("*.json"):
                if not file_path.name.endswith("_math.json"):
                    files.append({
                        "file_type": "easy",
                        "filename": file_path.name,
                        "original_filename": file_path.name,
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                        "user_id": None,
                        "created_at": datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
        
        if file_type in ["math", "all"]:
            for file_path in self.outputs_dir.glob("*_math.json"):
                files.append({
                    "file_type": "math",
                    "filename": file_path.name,
                    "original_filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "user_id": None,
                    "created_at": datetime.fromtimestamp(file_path.stat().st_mtime)
                })
        
        return sorted(files, key=lambda x: x["created_at"], reverse=True)
    
    def _sanitize_filename(self, name: str) -> str:
        """파일명 정리"""
        import re
        name = Path(name).stem  # 확장자 제거
        name = name.strip()[:200]
        return re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", name) or "doc"

# 전역 파일 관리자 인스턴스
file_manager = FileManager()


