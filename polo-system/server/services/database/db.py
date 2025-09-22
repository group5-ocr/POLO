from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Any, Dict

from sqlalchemy import (
    String, Integer, Boolean, ForeignKey, DateTime, Text, select, update, text, TIMESTAMP, JSON
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
)

# ---------------------------
# 1) ORM Base
# ---------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------
# 2) ORM Models
# ---------------------------
class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String, unique=True)
    password: Mapped[str] = mapped_column(String)
    nickname: Mapped[str] = mapped_column(String)
    job: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    create_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)

    origin_files = relationship("OriginFile", back_populates="user")


class OriginFile(Base):
    __tablename__ = "origin_file"
    origin_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    filename: Mapped[str] = mapped_column(String)
    create_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)
    id: Mapped[int] = mapped_column(Integer, nullable=True)

    user = relationship("User", back_populates="origin_files")
    tex_files = relationship("Tex", back_populates="origin_file")
    integrated_results = relationship("IntegratedResult", back_populates="origin_file")


class Tex(Base):
    __tablename__ = "tex"
    tex_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    origin_id: Mapped[int] = mapped_column(ForeignKey("origin_file.origin_id"))
    file_addr: Mapped[str] = mapped_column(String, nullable=True)

    # 진행 상태
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    easy_done: Mapped[int] = mapped_column(Integer, default=0)
    viz_done: Mapped[int] = mapped_column(Integer, default=0)
    math_done: Mapped[bool] = mapped_column(Boolean, default=False)
    integrated_done: Mapped[bool] = mapped_column(Boolean, default=False)
    integrated_result_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    origin_file = relationship("OriginFile", back_populates="tex_files")
    easy_files = relationship("EasyFile", back_populates="tex")
    math_files = relationship("MathFile", back_populates="tex")
    integrated_result = relationship("IntegratedResult", back_populates="tex")


class EasyFile(Base):
    __tablename__ = "easy_file"
    easy_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tex_id: Mapped[int] = mapped_column(ForeignKey("tex.tex_id"))
    origin_id: Mapped[int] = mapped_column(Integer)
    filename: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_addr: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    tex = relationship("Tex", back_populates="easy_files")


class MathFile(Base):
    __tablename__ = "math_file"
    math_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tex_id: Mapped[int] = mapped_column(ForeignKey("tex.tex_id"))
    origin_id: Mapped[int] = mapped_column(Integer)
    filename: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_addr: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    tex = relationship("Tex", back_populates="math_files")


class Chunk(Base):
    """
    paper_id = str(tex_id)
    key = f"{paper_id}:{index}"
    """
    __tablename__ = "chunks"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    paper_id: Mapped[str] = mapped_column(String, index=True)
    index: Mapped[int] = mapped_column(Integer)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rewritten_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    section_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class IntegratedResult(Base):
    __tablename__ = "integrated_results"
    result_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tex_id: Mapped[int] = mapped_column(ForeignKey("tex.tex_id"))
    origin_id: Mapped[int] = mapped_column(ForeignKey("origin_file.origin_id"))
    paper_id: Mapped[str] = mapped_column(String)
    paper_title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    paper_authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    paper_venue: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    total_sections: Mapped[int] = mapped_column(Integer, default=0)
    total_equations: Mapped[int] = mapped_column(Integer, default=0)
    total_visualizations: Mapped[int] = mapped_column(Integer, default=0)
    processing_status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)

    tex = relationship("Tex", back_populates="integrated_result")
    origin_file = relationship("OriginFile", back_populates="integrated_results")
    easy_sections = relationship("EasySection", back_populates="integrated_result", cascade="all, delete-orphan")
    math_equations = relationship("MathEquation", back_populates="integrated_result", cascade="all, delete-orphan")
    visualizations = relationship("Visualization", back_populates="integrated_result", cascade="all, delete-orphan")


class EasySection(Base):
    __tablename__ = "easy_sections"
    section_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    integrated_result_id: Mapped[int] = mapped_column(ForeignKey("integrated_results.result_id"))
    easy_section_id: Mapped[str] = mapped_column(String)
    easy_section_title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    easy_section_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    easy_section_order: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    easy_section_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    easy_section_parent: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    easy_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)

    integrated_result = relationship("IntegratedResult", back_populates="easy_sections")
    easy_paragraphs = relationship("EasyParagraph", back_populates="easy_section", cascade="all, delete-orphan")
    visualizations = relationship("Visualization", back_populates="easy_section")


class EasyParagraph(Base):
    __tablename__ = "easy_paragraphs"
    paragraph_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(ForeignKey("easy_sections.section_id"))
    easy_paragraph_id: Mapped[str] = mapped_column(String)
    easy_paragraph_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    easy_visualization_trigger: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)

    easy_section = relationship("EasySection", back_populates="easy_paragraphs")
    visualizations = relationship("Visualization", back_populates="easy_paragraph")


class MathEquation(Base):
    __tablename__ = "math_equations"
    equation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    integrated_result_id: Mapped[int] = mapped_column(ForeignKey("integrated_results.result_id"))
    math_equation_id: Mapped[str] = mapped_column(String)
    math_equation_latex: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    math_equation_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    math_equation_section_ref: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)

    integrated_result = relationship("IntegratedResult", back_populates="math_equations")


class Visualization(Base):
    __tablename__ = "visualizations"
    viz_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    integrated_result_id: Mapped[int] = mapped_column(ForeignKey("integrated_results.result_id"))
    section_id: Mapped[Optional[int]] = mapped_column(ForeignKey("easy_sections.section_id"), nullable=True)
    paragraph_id: Mapped[Optional[int]] = mapped_column(ForeignKey("easy_paragraphs.paragraph_id"), nullable=True)
    viz_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    viz_title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    viz_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    viz_image_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    viz_metadata: Mapped[Optional[dict]] = mapped_column(Text, nullable=True)  # SQLite/PostgreSQL 호환
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), default=datetime.utcnow)

    integrated_result = relationship("IntegratedResult", back_populates="visualizations")
    easy_section = relationship("EasySection", back_populates="visualizations")
    easy_paragraph = relationship("EasyParagraph", back_populates="visualizations")


# ---------------------------
# 3) DB Router (PG ↔ SQLite)
# ---------------------------
class DBRouter:
    def __init__(self) -> None:
        self.engine: Optional[AsyncEngine] = None
        self._session: Optional[async_sessionmaker[AsyncSession]] = None
        self.mode: str = "local"  # "pg" or "local"

    async def init(self) -> None:
        # 환경변수 디버깅
        postgres_user = os.getenv('POSTGRES_USER')
        postgres_password = os.getenv('POSTGRES_PASSWORD')
        postgres_host = os.getenv('POSTGRES_HOST')
        postgres_port = os.getenv('POSTGRES_PORT', '5432')
        postgres_db = os.getenv('POSTGRES_DB')
        
        print(f"[DEBUG] PostgreSQL 환경변수:")
        print(f"  POSTGRES_USER: {postgres_user}")
        print(f"  POSTGRES_PASSWORD: {'*' * len(postgres_password) if postgres_password else 'None'}")
        print(f"  POSTGRES_HOST: {postgres_host}")
        print(f"  POSTGRES_PORT: {postgres_port}")
        print(f"  POSTGRES_DB: {postgres_db}")
        
        # 절대 경로로 DB 경로 설정
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent.parent  # polo-system/server
        default_db_path = server_dir / "data" / "local" / "polo.db"
        local_path = os.getenv("LOCAL_DB_PATH", str(default_db_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite+aiosqlite:///{local_path}"

        # PostgreSQL 연결 시도
        if all([postgres_user, postgres_password, postgres_host, postgres_db]):
            pg_url = (
                f"postgresql+asyncpg://{postgres_user}:{postgres_password}"
                f"@{postgres_host}:{postgres_port}/{postgres_db}"
            )
            print(f"[DEBUG] PostgreSQL URL: {pg_url}")

            try:
                print(f"[DEBUG] PostgreSQL 연결 시도 중...")
                pg_engine = create_async_engine(pg_url, pool_pre_ping=True)
                await asyncio.wait_for(self._ping(pg_engine), timeout=5.0)
                self.engine = pg_engine
                self.mode = "pg"
                print("✅ PostgreSQL 연결 성공")
            except Exception as e:
                print(f"❌ PostgreSQL 연결 실패: {str(e)}")
                print(f"❌ 오류 타입: {type(e).__name__}")
                self.engine = create_async_engine(sqlite_url)
                self.mode = "local"
                print("⚠️ PostgreSQL 연결 실패 → SQLite 사용")
        else:
            print("⚠️ PostgreSQL 환경변수가 설정되지 않음 → SQLite 사용")
            self.engine = create_async_engine(sqlite_url)
            self.mode = "local"

        self._session = async_sessionmaker(self.engine, expire_on_commit=False)

        # 테이블 생성 (PostgreSQL과 SQLite 모두에서 작동)
        try:
            async with self.engine.begin() as conn:
                # PostgreSQL에서는 JSONB 확장 활성화
                if self.mode == "pg":
                    try:
                        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                        print("✅ PostgreSQL 확장 활성화 완료")
                    except Exception as ext_e:
                        print(f"⚠️ 확장 활성화 실패 (무시 가능): {ext_e}")
                
                # 테이블 생성
                await conn.run_sync(Base.metadata.create_all)
                
                # SQLite에서는 추가 제약조건 설정
                if self.mode == "local":
                    try:
                        # SQLite에서 외래키 활성화
                        await conn.execute(text("PRAGMA foreign_keys = ON"))
                        print("✅ SQLite 외래키 활성화 완료")
                    except Exception as fk_e:
                        print(f"⚠️ SQLite 외래키 활성화 실패: {fk_e}")
                
            print(f"✅ 테이블 생성 완료 ({self.mode} 모드)")
        except Exception as e:
            print(f"❌ 테이블 생성 실패: {e}")
            raise

    async def _ping(self, engine: AsyncEngine) -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    def session(self) -> async_sessionmaker[AsyncSession]:
        if not self._session:
            raise RuntimeError("DB 초기화 필요")
        return self._session
    
    async def test_connection(self) -> bool:
        """데이터베이스 연결 테스트"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"❌ DB 연결 테스트 실패: {e}")
            return False
    
    async def get_table_info(self) -> Dict[str, Any]:
        """데이터베이스 테이블 정보 조회"""
        try:
            async with self.engine.connect() as conn:
                if self.mode == "pg":
                    # PostgreSQL
                    result = await conn.execute(text("""
                        SELECT table_name, column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        ORDER BY table_name, ordinal_position
                    """))
                    tables = {}
                    for row in result:
                        table_name = row[0]
                        if table_name not in tables:
                            tables[table_name] = []
                        tables[table_name].append({
                            "column": row[1],
                            "type": row[2],
                            "nullable": row[3] == "YES"
                        })
                else:
                    # SQLite
                    result = await conn.execute(text("""
                        SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
                    """))
                    tables = {}
                    for row in result:
                        table_name = row[0]
                        # 각 테이블의 컬럼 정보 조회
                        col_result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
                        tables[table_name] = []
                        for col_row in col_result:
                            tables[table_name].append({
                                "column": col_row[1],
                                "type": col_row[2],
                                "nullable": col_row[3] == 0
                            })
                
                return {
                    "mode": self.mode,
                    "tables": tables,
                    "total_tables": len(tables)
                }
        except Exception as e:
            print(f"❌ 테이블 정보 조회 실패: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    # CRUD 메서드들
    async def create_origin_file(self, user_id: int, filename: str) -> int:
        return await create_origin_file(user_id, filename)

    async def create_tex(self, origin_id: int, file_addr: str) -> int:
        return await create_tex(origin_id, file_addr)
    
    async def get_origin_id_from_tex(self, tex_id: int) -> Optional[int]:
        return await get_origin_id_from_tex(tex_id)
    
    async def save_easy_chunk(self, tex_id: int | str, index: int, rewritten_text: str) -> None:
        return await save_easy_chunk(tex_id, index, rewritten_text)
    
    async def save_easy_result(self, tex_id: int, origin_id: int, result_path: str) -> None:
        return await save_easy_result(tex_id, origin_id, result_path)
    
    async def save_math_result(self, tex_id: int, origin_id: int, result_path: str, sections: Optional[List[Dict[str, Any]]] = None) -> None:
        return await save_math_result(tex_id, origin_id, result_path, sections)
    
    async def set_flag(self, tex_id: int, field: str, value: Any) -> None:
        return await set_flag(tex_id, field, value)
    
    async def get_state(self, tex_id: int) -> Optional[Tex]:
        return await get_state(tex_id)


DB = DBRouter()


# ---------------------------
# 4) CRUD / 상태 관리
# ---------------------------
async def create_origin_file(user_id: int, filename: str) -> int:
    async with DB.session()() as s:
        origin = OriginFile(user_id=user_id, filename=filename, id=user_id)
        s.add(origin)
        await s.commit()
        await s.refresh(origin)
        return origin.origin_id


async def create_tex(origin_id: int, file_addr: str) -> int:
    async with DB.session()() as s:
        tex = Tex(origin_id=origin_id, file_addr=file_addr)
        s.add(tex)
        await s.commit()
        await s.refresh(tex)
        return tex.tex_id


async def create_easy_file(tex_id: int, origin_id: int, filename: str, file_addr: str) -> int:
    async with DB.session()() as s:
        easy = EasyFile(tex_id=tex_id, origin_id=origin_id, filename=filename, file_addr=file_addr)
        s.add(easy)
        await s.commit()
        await s.refresh(easy)
        return easy.easy_id


# ---- 파이프라인 상태 ----
async def init_pipeline_state(tex_id: int, total_chunks: int, jsonl_path: str, math_text_path: str) -> None:
    async with DB.session()() as s:
        await s.execute(
            update(Tex)
            .where(Tex.tex_id == tex_id)
            .values(total_chunks=total_chunks, easy_done=0, viz_done=0, math_done=False)
        )
        await s.commit()


async def bump_counter(tex_id: int, field: str) -> None:
    if field not in ("easy_done", "viz_done"):
        raise ValueError("field는 'easy_done' 또는 'viz_done'만 허용")
    async with DB.session()() as s:
        await s.execute(
            update(Tex)
            .where(Tex.tex_id == tex_id)
            .values(**{field: getattr(Tex, field) + 1})
        )
        await s.commit()


async def set_flag(tex_id: int, field: str, value: bool) -> None:
    if field not in ("math_done",):
        raise ValueError("field는 'math_done'만 허용")
    async with DB.session()() as s:
        await s.execute(
            update(Tex).where(Tex.tex_id == tex_id).values(**{field: value})
        )
        await s.commit()


async def get_state(tex_id: int) -> Optional[Tex]:
    async with DB.session()() as s:
        res = await s.execute(select(Tex).where(Tex.tex_id == tex_id))
        return res.scalar_one_or_none()


async def get_origin_id_from_tex(tex_id: int) -> Optional[int]:
    async with DB.session()() as s:
        res = await s.execute(select(Tex.origin_id).where(Tex.tex_id == tex_id))
        return res.scalar_one_or_none()


# ---- Chunk / Easy / Viz / Math ----
async def save_easy_chunk(tex_id: int | str, index: int, rewritten_text: str) -> None:
    paper_id = str(tex_id)
    key = f"{paper_id}:{index}"
    async with DB.session()() as s:
        chunk = await s.get(Chunk, key)
        if not chunk:
            chunk = Chunk(key=key, paper_id=paper_id, index=index, rewritten_text=rewritten_text)
            s.add(chunk)
        else:
            chunk.rewritten_text = rewritten_text
        await s.commit()


async def save_viz_image(tex_id: int | str, index: int, image_path: str) -> None:
    paper_id = str(tex_id)
    key = f"{paper_id}:{index}"
    async with DB.session()() as s:
        chunk = await s.get(Chunk, key)
        if not chunk:
            chunk = Chunk(key=key, paper_id=paper_id, index=index, image_path=image_path)
            s.add(chunk)
        else:
            chunk.image_path = image_path
        await s.commit()


async def save_easy_result(tex_id: int, origin_id: int, result_path: str) -> int:
    async with DB.session()() as s:
        easy = EasyFile(tex_id=tex_id, origin_id=origin_id, filename="easy_results.json", file_addr=result_path)
        s.add(easy)
        await s.commit()
        await s.refresh(easy)
        return easy.easy_id


async def save_math_result(
    tex_id: int,
    origin_id: int,
    result_path: Optional[str] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> int:
    async with DB.session()() as s:
        math = MathFile(tex_id=tex_id, origin_id=origin_id, file_addr=result_path)
        s.add(math)
        await s.commit()
        await s.refresh(math)
        return math.math_id


# ---- 결과 조회 & 조립 ----
async def fetch_results(tex_id: int | str) -> Optional[Dict[str, Any]]:
    paper_id = str(tex_id)
    async with DB.session()() as s:
        # chunks
        chunk_rows = await s.execute(
            select(Chunk).where(Chunk.paper_id == paper_id).order_by(Chunk.index)
        )
        chunks = chunk_rows.scalars().all()
        chunk_list = [
            {
                "index": c.index,
                "text": c.text,
                "rewritten_text": c.rewritten_text,
                "image_path": c.image_path,
            }
            for c in chunks
        ]

        # math files
        math_rows = await s.execute(select(MathFile).where(MathFile.tex_id == int(paper_id)))
        math_files = [
            {"math_id": m.math_id, "origin_id": m.origin_id, "file_addr": m.file_addr}
            for m in math_rows.scalars().all()
        ]
        
        # easy files
        easy_rows = await s.execute(select(EasyFile).where(EasyFile.tex_id == int(paper_id)))
        easy_files = [
            {"easy_id": e.easy_id, "origin_id": e.origin_id, "filename": e.filename, "file_addr": e.file_addr}
            for e in easy_rows.scalars().all()
        ]

        # 상태
        st = await get_state(int(paper_id))
        if not st:
            return None

        return {
            "paper_id": int(paper_id),
            "total_chunks": st.total_chunks,
            "easy_done": st.easy_done,
            "viz_done": st.viz_done,
            "math_done": st.math_done,
            "items": chunk_list,
            "math": {"files": math_files},
            "easy": {"files": easy_files},
        }


async def assemble_final(tex_id: int) -> Dict[str, Any]:
    data = await fetch_results(tex_id)
    return data if data else {"paper_id": tex_id, "items": [], "math": {"files": []}, "easy": {"files": []}}


# ---- 통합 결과 관리 함수들 ----
async def create_integrated_result(
    tex_id: int,
    origin_id: int,
    paper_id: str,
    paper_title: Optional[str] = None,
    paper_authors: Optional[str] = None,
    paper_venue: Optional[str] = None,
    processing_status: str = "pending"
) -> int:
    """통합 결과 생성"""
    async with DB.session()() as s:
        result = IntegratedResult(
            tex_id=tex_id,
            origin_id=origin_id,
            paper_id=paper_id,
            paper_title=paper_title,
            paper_authors=paper_authors,
            paper_venue=paper_venue,
            processing_status=processing_status
        )
        s.add(result)
        await s.commit()
        await s.refresh(result)
        return result.result_id


async def update_integrated_result(
    result_id: int,
    paper_title: Optional[str] = None,
    paper_authors: Optional[str] = None,
    paper_venue: Optional[str] = None,
    total_sections: Optional[int] = None,
    total_equations: Optional[int] = None,
    total_visualizations: Optional[int] = None,
    processing_status: Optional[str] = None
) -> bool:
    """통합 결과 업데이트"""
    async with DB.session()() as s:
        result = await s.get(IntegratedResult, result_id)
        if not result:
            return False
        
        if paper_title is not None:
            result.paper_title = paper_title
        if paper_authors is not None:
            result.paper_authors = paper_authors
        if paper_venue is not None:
            result.paper_venue = paper_venue
        if total_sections is not None:
            result.total_sections = total_sections
        if total_equations is not None:
            result.total_equations = total_equations
        if total_visualizations is not None:
            result.total_visualizations = total_visualizations
        if processing_status is not None:
            result.processing_status = processing_status
        
        result.updated_at = datetime.utcnow()
        await s.commit()
        return True


async def save_easy_sections(
    integrated_result_id: int,
    sections: List[Dict[str, Any]]
) -> List[int]:
    """Easy 섹션들 저장"""
    section_ids = []
    async with DB.session()() as s:
        for section_data in sections:
            section = EasySection(
                integrated_result_id=integrated_result_id,
                easy_section_id=section_data.get("easy_section_id", ""),
                easy_section_title=section_data.get("easy_section_title"),
                easy_section_type=section_data.get("easy_section_type"),
                easy_section_order=section_data.get("easy_section_order"),
                easy_section_level=section_data.get("easy_section_level"),
                easy_section_parent=section_data.get("easy_section_parent"),
                easy_content=section_data.get("easy_content")
            )
            s.add(section)
            await s.flush()  # ID를 얻기 위해 flush
            section_ids.append(section.section_id)
            
            # 문단들 저장
            if "easy_paragraphs" in section_data:
                for paragraph_data in section_data["easy_paragraphs"]:
                    paragraph = EasyParagraph(
                        section_id=section.section_id,
                        easy_paragraph_id=paragraph_data.get("easy_paragraph_id", ""),
                        easy_paragraph_text=paragraph_data.get("easy_paragraph_text"),
                        easy_visualization_trigger=paragraph_data.get("easy_visualization_trigger", False)
                    )
                    s.add(paragraph)
        
        await s.commit()
        return section_ids


async def save_math_equations(
    integrated_result_id: int,
    equations: List[Dict[str, Any]]
) -> List[int]:
    """Math 수식들 저장"""
    equation_ids = []
    async with DB.session()() as s:
        for equation_data in equations:
            equation = MathEquation(
                integrated_result_id=integrated_result_id,
                math_equation_id=equation_data.get("math_equation_id", ""),
                math_equation_latex=equation_data.get("math_equation_latex"),
                math_equation_explanation=equation_data.get("math_equation_explanation"),
                math_equation_section_ref=equation_data.get("math_equation_section_ref")
            )
            s.add(equation)
            await s.flush()
            equation_ids.append(equation.equation_id)
        
        await s.commit()
        return equation_ids


async def save_visualizations(
    integrated_result_id: int,
    visualizations: List[Dict[str, Any]]
) -> List[int]:
    """시각화들 저장"""
    viz_ids = []
    async with DB.session()() as s:
        for viz_data in visualizations:
            viz = Visualization(
                integrated_result_id=integrated_result_id,
                section_id=viz_data.get("section_id"),
                paragraph_id=viz_data.get("paragraph_id"),
                viz_type=viz_data.get("viz_type"),
                viz_title=viz_data.get("viz_title"),
                viz_description=viz_data.get("viz_description"),
                viz_image_path=viz_data.get("viz_image_path"),
                viz_metadata=json.dumps(viz_data.get("viz_metadata")) if viz_data.get("viz_metadata") else None
            )
            s.add(viz)
            await s.flush()
            viz_ids.append(viz.viz_id)
        
        await s.commit()
        return viz_ids


async def get_integrated_result(tex_id: int) -> Optional[Dict[str, Any]]:
    """통합 결과 조회"""
    async with DB.session()() as s:
        # 통합 결과 조회
        result = await s.execute(
            select(IntegratedResult).where(IntegratedResult.tex_id == tex_id)
        )
        integrated_result = result.scalar_one_or_none()
        
        if not integrated_result:
            return None
        
        # Easy 섹션들 조회
        sections_result = await s.execute(
            select(EasySection).where(EasySection.integrated_result_id == integrated_result.result_id)
            .order_by(EasySection.easy_section_order)
        )
        sections = sections_result.scalars().all()
        
        easy_sections = []
        for section in sections:
            # 문단들 조회
            paragraphs_result = await s.execute(
                select(EasyParagraph).where(EasyParagraph.section_id == section.section_id)
            )
            paragraphs = paragraphs_result.scalars().all()
            
            # 시각화들 조회
            viz_result = await s.execute(
                select(Visualization).where(Visualization.section_id == section.section_id)
            )
            visualizations = viz_result.scalars().all()
            
            section_data = {
                "easy_section_id": section.easy_section_id,
                "easy_section_title": section.easy_section_title,
                "easy_section_type": section.easy_section_type,
                "easy_section_order": section.easy_section_order,
                "easy_section_level": section.easy_section_level,
                "easy_section_parent": section.easy_section_parent,
                "easy_content": section.easy_content,
                "easy_paragraphs": [
                    {
                        "easy_paragraph_id": p.easy_paragraph_id,
                        "easy_paragraph_text": p.easy_paragraph_text,
                        "easy_visualization_trigger": p.easy_visualization_trigger
                    } for p in paragraphs
                ],
                "easy_visualizations": [
                    {
                        "viz_id": v.viz_id,
                        "viz_type": v.viz_type,
                        "viz_title": v.viz_title,
                        "viz_description": v.viz_description,
                        "viz_image_path": v.viz_image_path,
                        "viz_metadata": json.loads(v.viz_metadata) if v.viz_metadata else None
                    } for v in visualizations
                ]
            }
            easy_sections.append(section_data)
        
        # Math 수식들 조회
        equations_result = await s.execute(
            select(MathEquation).where(MathEquation.integrated_result_id == integrated_result.result_id)
        )
        equations = equations_result.scalars().all()
        
        math_equations = [
            {
                "math_equation_id": e.math_equation_id,
                "math_equation_latex": e.math_equation_latex,
                "math_equation_explanation": e.math_equation_explanation,
                "math_equation_section_ref": e.math_equation_section_ref
            } for e in equations
        ]
        
        return {
            "paper_info": {
                "paper_id": integrated_result.paper_id,
                "paper_title": integrated_result.paper_title,
                "paper_authors": integrated_result.paper_authors,
                "paper_venue": integrated_result.paper_venue,
                "total_sections": integrated_result.total_sections,
                "total_equations": integrated_result.total_equations,
                "total_visualizations": integrated_result.total_visualizations
            },
            "easy_sections": easy_sections,
            "math_equations": math_equations,
            "processing_status": integrated_result.processing_status,
            "created_at": integrated_result.created_at.isoformat(),
            "updated_at": integrated_result.updated_at.isoformat()
        }


# ---- 통합 결과 자동 생성 함수 ----
async def create_integrated_result_from_files(
    tex_id: int,
    paper_id: str,
    easy_data: Optional[Dict[str, Any]] = None,
    math_data: Optional[Dict[str, Any]] = None,
    viz_data: Optional[Dict[str, Any]] = None
) -> int:
    """파일 데이터로부터 통합 결과 자동 생성"""
    try:
        # origin_id 찾기
        async with DB.session()() as s:
            tex_result = await s.execute(select(Tex).where(Tex.tex_id == tex_id))
            tex_record = tex_result.scalar_one_or_none()
            if not tex_record:
                raise ValueError(f"Tex record not found for tex_id: {tex_id}")
            origin_id = tex_record.origin_id
        
        # 통합 결과 생성
        result_id = await create_integrated_result(
            tex_id=tex_id,
            origin_id=origin_id,
            paper_id=paper_id,
            processing_status="processing"
        )
        
        # Easy 데이터 저장
        if easy_data and "easy_sections" in easy_data:
            section_ids = await save_easy_sections(result_id, easy_data["easy_sections"])
            print(f"✅ [DB] Easy 섹션 {len(section_ids)}개 저장 완료")
            
            # 논문 정보 업데이트
            paper_info = easy_data.get("paper_info", {})
            await update_integrated_result(
                result_id=result_id,
                paper_title=paper_info.get("paper_title"),
                paper_authors=paper_info.get("paper_authors"),
                paper_venue=paper_info.get("paper_venue"),
                total_sections=len(easy_data.get("easy_sections", []))
            )
        
        # Math 데이터 저장
        if math_data and "items" in math_data:
            equations = []
            for i, item in enumerate(math_data["items"]):
                equation = {
                    "math_equation_id": f"eq_{i+1}",
                    "math_equation_latex": item.get("latex"),
                    "math_equation_explanation": item.get("explanation"),
                    "math_equation_section_ref": item.get("section_ref")
                }
                equations.append(equation)
            
            equation_ids = await save_math_equations(result_id, equations)
            print(f"✅ [DB] Math 수식 {len(equation_ids)}개 저장 완료")
            
            await update_integrated_result(
                result_id=result_id,
                total_equations=len(equations)
            )
        
        # Viz 데이터 저장
        if viz_data and "generated_visualizations" in viz_data:
            visualizations = []
            for viz in viz_data["generated_visualizations"]:
                viz_item = {
                    "viz_type": viz.get("visualization_type"),
                    "viz_title": viz.get("title"),
                    "viz_description": viz.get("description"),
                    "viz_image_path": viz.get("image_path"),
                    "viz_metadata": viz.get("metadata")
                }
                visualizations.append(viz_item)
            
            viz_ids = await save_visualizations(result_id, visualizations)
            print(f"✅ [DB] 시각화 {len(viz_ids)}개 저장 완료")
            
            await update_integrated_result(
                result_id=result_id,
                total_visualizations=len(visualizations)
            )
        
        # 완료 상태로 업데이트
        await update_integrated_result(
            result_id=result_id,
            processing_status="completed"
        )
        
        # tex 테이블 업데이트
        async with DB.session()() as s:
            tex_record.integrated_done = True
            tex_record.integrated_result_id = result_id
            await s.commit()
        
        print(f"✅ [DB] 통합 결과 생성 완료: result_id={result_id}")
        return result_id
        
    except Exception as e:
        print(f"❌ [DB] 통합 결과 생성 실패: {e}")
        raise


async def migrate_file_results_to_db(paper_id: str) -> bool:
    """기존 파일 기반 결과를 데이터베이스로 마이그레이션"""
    try:
        tex_id = int(paper_id)
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent.parent  # polo-system/server
        output_dir = server_dir / "data" / "outputs" / paper_id
        
        # Easy 결과 로드
        easy_data = None
        easy_file = output_dir / "easy_outputs" / "easy_results.json"
        if easy_file.exists():
            with open(easy_file, 'r', encoding='utf-8') as f:
                easy_data = json.load(f)
        
        # Math 결과 로드
        math_data = None
        math_file = output_dir / "math_outputs" / "equations_explained.json"
        if math_file.exists():
            with open(math_file, 'r', encoding='utf-8') as f:
                math_data = json.load(f)
        
        # Viz 결과 로드
        viz_data = None
        viz_file = output_dir / "viz_outputs" / "visualizations.json"
        if viz_file.exists():
            with open(viz_file, 'r', encoding='utf-8') as f:
                viz_data = json.load(f)
        
        # 통합 결과 생성
        result_id = await create_integrated_result_from_files(
            tex_id=tex_id,
            paper_id=paper_id,
            easy_data=easy_data,
            math_data=math_data,
            viz_data=viz_data
        )
        
        print(f"✅ [DB] 마이그레이션 완료: paper_id={paper_id}, result_id={result_id}")
        return True
        
    except Exception as e:
        print(f"❌ [DB] 마이그레이션 실패: {e}")
        return False


# ---------------------------
# 5) 모듈 레벨 프록시 (안전장치)
# ---------------------------
# 모듈을 `import services.database.db as DB`로 임포트했을 때도
# DB.init(), DB.session(), DB.close()가 동작하도록 프록시를 둡니다.
async def init():               # type: ignore
    await DB.init()

def session():                  # type: ignore
    return DB.session()

async def close():              # type: ignore
    await DB.close()
