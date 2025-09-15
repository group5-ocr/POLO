# server/routes/database.py
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.database.db import DB, User

router = APIRouter()

# ==== 스키마 ====
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    nickname: Optional[str] = None
    job: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    user_id: int
    email: str
    nickname: str
    job: Optional[str] = None
    create_at: datetime

class LoginResponse(BaseModel):
    user: UserResponse
    token: str

class UsersResponse(BaseModel):
    users: List[UserResponse]

# ==== 유틸리티 함수 ====
def hash_password(password: str) -> str:
    """비밀번호 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    return hash_password(password) == hashed

# ==== 라우트 ====
@router.post("/users", response_model=UserResponse)
async def create_user(user_data: UserCreate):
    """
    회원가입
    """
    try:
        # 이메일 중복 확인
        async with DB.session() as session:
            result = await session.execute(
                select(User).where(User.email == user_data.email)
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")
            
            # 새 사용자 생성
            hashed_password = hash_password(user_data.password)
            new_user = User(
                email=user_data.email,
                password=hashed_password,
                nickname=user_data.nickname or user_data.username,
                job=user_data.job
            )
            
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            
            return UserResponse(
                user_id=new_user.user_id,
                email=new_user.email,
                nickname=new_user.nickname,
                job=new_user.job,
                create_at=new_user.create_at
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"회원가입 중 오류가 발생했습니다: {str(e)}")

@router.post("/login", response_model=LoginResponse)
async def login(login_data: UserLogin):
    """
    로그인
    """
    try:
        async with DB.session() as session:
            # 사용자 조회
            result = await session.execute(
                select(User).where(User.email == login_data.email)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
            
            # 비밀번호 확인
            if not verify_password(login_data.password, user.password):
                raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
            
            # 간단한 토큰 생성 (실제로는 JWT 사용 권장)
            token = f"token_{user.user_id}_{datetime.now().timestamp()}"
            
            return LoginResponse(
                user=UserResponse(
                    user_id=user.user_id,
                    email=user.email,
                    nickname=user.nickname,
                    job=user.job,
                    create_at=user.create_at
                ),
                token=token
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"로그인 중 오류가 발생했습니다: {str(e)}")

@router.get("/users", response_model=UsersResponse)
async def get_users():
    """
    사용자 목록 조회 (로그인 상태 확인용)
    """
    try:
        async with DB.session() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            
            user_list = [
                UserResponse(
                    user_id=user.user_id,
                    email=user.email,
                    nickname=user.nickname,
                    job=user.job,
                    create_at=user.create_at
                )
                for user in users
            ]
            
            return UsersResponse(users=user_list)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사용자 조회 중 오류가 발생했습니다: {str(e)}")

@router.get("/stats")
async def get_database_stats():
    """
    데이터베이스 통계 (간단한 버전)
    """
    try:
        async with DB.session() as session:
            # 사용자 수 조회
            result = await session.execute(select(User))
            users = result.scalars().all()
            user_count = len(users)
            
            return {
                "database_stats": {
                    "users": user_count,
                    "documents": 0,  # 나중에 구현
                    "processing_jobs": 0,  # 나중에 구현
                    "model_results": 0,  # 나중에 구현
                    "jobs_by_status": {}  # 나중에 구현
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}")

