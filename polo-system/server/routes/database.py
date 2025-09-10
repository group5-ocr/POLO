# server/routes/database.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime

from services.db import get_database, get_postgres_client, PostgreSQLClient
from utils.password import hash_password, verify_password

router = APIRouter()

# ===== SQLAlchemy ORM 사용 예제 =====

@router.get("/users")
async def get_users(db: Session = Depends(get_database)):
    """모든 사용자 조회 (SQLAlchemy ORM 사용)"""
    try:
        from services.db import Base
        from sqlalchemy import text
        
        # 직접 SQL 쿼리 실행
        result = db.execute(text("SELECT user_id, email, password, nickname, job, create_at FROM users ORDER BY create_at DESC"))
        users = []
        for row in result:
            users.append({
                "user_id": row[0],
                "email": row[1],
                "password": row[2],
                "nickname": row[3],
                "job": row[4],
                "created_at": row[5].isoformat() if row[5] else None
            })
        
        return {"users": users, "count": len(users)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용자 조회 실패: {str(e)}"
        )

@router.post("/users")
async def create_user(
    request: dict,
    db: Session = Depends(get_database)
):
    """새 사용자 생성 (SQLAlchemy ORM 사용)"""
    try:
        from sqlalchemy import text
        
        # 요청 데이터 추출
        username = request.get("username", "")
        email = request.get("email", "")
        password = request.get("password", "")
        
        # 필수 필드 검증
        if not email or not password or not username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이메일, 비밀번호, 사용자명은 필수입니다."
            )
        
        # 사용자 존재 여부 확인
        check_result = db.execute(
            text("SELECT user_id FROM users WHERE email = :email"),
            {"email": email}
        )
        if check_result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이메일이 이미 존재합니다."
            )
        
        # 비밀번호 해시화
        hashed_password = hash_password(password)
        
        # 새 사용자 생성
        result = db.execute(
            text("""
                INSERT INTO users (email, password, nickname, job, create_at)
                VALUES (:email, :password, :nickname, :job, :create_at)
                RETURNING user_id
            """),
            {
                "email": email,
                "password": hashed_password,  # 해시화된 비밀번호 저장
                "nickname": username,  # username을 nickname으로 사용
                "job": "일반사용자",  # 기본값
                "create_at": datetime.now()
            }
        )
        
        user_id = result.fetchone()[0]
        db.commit()
        
        return {
            "message": "사용자가 성공적으로 생성되었습니다.",
            "user_id": user_id,
            "nickname": username,
            "email": email
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"사용자 생성 실패: {str(e)}"
        )

@router.post("/login")
async def login_user(
    request: dict,
    db: Session = Depends(get_database)
):
    """사용자 로그인 (비밀번호 검증)"""
    try:
        from sqlalchemy import text
        
        # 요청 데이터 추출
        email = request.get("email", "")
        password = request.get("password", "")
        
        # 필수 필드 검증
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이메일과 비밀번호는 필수입니다."
            )
        
        # 사용자 조회
        result = db.execute(
            text("SELECT user_id, email, password, nickname, job, create_at FROM users WHERE email = :email"),
            {"email": email}
        )
        user_row = result.fetchone()
        
        if not user_row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )
        
        # 비밀번호 검증
        stored_password = user_row[2]
        if not verify_password(password, stored_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다."
            )
        
        # 로그인 성공 - 사용자 정보 반환 (비밀번호 제외)
        return {
            "message": "로그인 성공",
            "user": {
                "user_id": user_row[0],
                "email": user_row[1],
                "nickname": user_row[3],
                "job": user_row[4],
                "created_at": user_row[5].isoformat() if user_row[5] else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"로그인 실패: {str(e)}"
        )

# ===== psycopg2 직접 사용 예제 =====

# 문서 및 작업 관련 API는 현재 사용하지 않으므로 제거

@router.get("/stats")
async def get_database_stats(pg_client: PostgreSQLClient = Depends(get_postgres_client)):
    """데이터베이스 통계 정보 조회"""
    try:
        stats = {}
        
        # 사용자 수
        user_count = pg_client.execute_query("SELECT COUNT(*) as count FROM users")
        stats["users"] = user_count[0]["count"] if user_count else 0
        
        # 문서 수 (origin_file 테이블 사용)
        doc_count = pg_client.execute_query("SELECT COUNT(*) as count FROM origin_file")
        stats["documents"] = doc_count[0]["count"] if doc_count else 0
        
        # 처리 작업 수 (현재 사용 안함)
        stats["processing_jobs"] = 0
        
        # 모델 결과 수 (현재 사용 안함)
        stats["model_results"] = 0
        
        # 상태별 작업 수 (현재 사용 안함)
        stats["jobs_by_status"] = {}
        
        return {"database_stats": stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"통계 조회 실패: {str(e)}"
        )
