from __future__ import annotations

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Any, Dict

from sqlalchemy import (
    String, Integer, Boolean, ForeignKey, DateTime, Text, select, update, text
)
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
    create_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    origin_files = relationship("OriginFile", back_populates="user")


class OriginFile(Base):
    __tablename__ = "origin_file"
    origin_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    filename: Mapped[str] = mapped_column(String)

    user = relationship("User", back_populates="origin_files")
    tex_files = relationship("Tex", back_populates="origin_file")


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

    origin_file = relationship("OriginFile", back_populates="tex_files")
    easy_files = relationship("EasyFile", back_populates="tex")
    math_files = relationship("MathFile", back_populates="tex")


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


# ---------------------------
# 3) DB Router (PG ↔ SQLite)
# ---------------------------
class DBRouter:
    def __init__(self) -> None:
        self.engine: Optional[AsyncEngine] = None
        self._session: Optional[async_sessionmaker[AsyncSession]] = None
        self.mode: str = "local"  # "pg" or "local"

    async def init(self) -> None:
        pg_url = (
            f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
        )

        # 절대 경로로 DB 경로 설정
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent.parent.parent  # polo-system/server
        default_db_path = server_dir / "data" / "local" / "polo.db"
        local_path = os.getenv("LOCAL_DB_PATH", str(default_db_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite+aiosqlite:///{local_path}"

        try:
            pg_engine = create_async_engine(pg_url, pool_pre_ping=True)
            await asyncio.wait_for(self._ping(pg_engine), timeout=2.0)
            self.engine = pg_engine
            self.mode = "pg"
            print("✅ PostgreSQL 연결 성공")
        except Exception:
            self.engine = create_async_engine(sqlite_url)
            self.mode = "local"
            print("⚠️ PostgreSQL 연결 실패 → SQLite 사용")

        self._session = async_sessionmaker(self.engine, expire_on_commit=False)

        # 테이블 생성
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _ping(self, engine: AsyncEngine) -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    def session(self) -> async_sessionmaker[AsyncSession]:
        if not self._session:
            raise RuntimeError("DB 초기화 필요")
        return self._session()

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    # CRUD 메서드들
    async def create_origin_file(self, user_id: int, filename: str) -> int:
        return await create_origin_file(user_id, filename)

    async def create_tex(self, origin_id: int, file_addr: str) -> int:
        return await create_tex(origin_id, file_addr)


DB = DBRouter()


# ---------------------------
# 4) CRUD / 상태 관리
# ---------------------------
async def create_origin_file(user_id: int, filename: str) -> int:
    async with DB.session() as s:
        origin = OriginFile(user_id=user_id, filename=filename)
        s.add(origin)
        await s.commit()
        await s.refresh(origin)
        return origin.origin_id


async def create_tex(origin_id: int, file_addr: str) -> int:
    async with DB.session() as s:
        tex = Tex(origin_id=origin_id, file_addr=file_addr)
        s.add(tex)
        await s.commit()
        await s.refresh(tex)
        return tex.tex_id


# ---- 파이프라인 상태 ----
async def init_pipeline_state(tex_id: int, total_chunks: int, jsonl_path: str, math_text_path: str) -> None:
    async with DB.session() as s:
        await s.execute(
            update(Tex)
            .where(Tex.tex_id == tex_id)
            .values(total_chunks=total_chunks, easy_done=0, viz_done=0, math_done=False)
        )
        await s.commit()


async def bump_counter(tex_id: int, field: str) -> None:
    if field not in ("easy_done", "viz_done"):
        raise ValueError("field는 'easy_done' 또는 'viz_done'만 허용")
    async with DB.session() as s:
        await s.execute(
            update(Tex)
            .where(Tex.tex_id == tex_id)
            .values(**{field: getattr(Tex, field) + 1})
        )
        await s.commit()


async def set_flag(tex_id: int, field: str, value: bool) -> None:
    if field not in ("math_done",):
        raise ValueError("field는 'math_done'만 허용")
    async with DB.session() as s:
        await s.execute(
            update(Tex).where(Tex.tex_id == tex_id).values(**{field: value})
        )
        await s.commit()


async def get_state(tex_id: int) -> Optional[Tex]:
    async with DB.session() as s:
        res = await s.execute(select(Tex).where(Tex.tex_id == tex_id))
        return res.scalar_one_or_none()


# ---- Chunk / Easy / Viz / Math ----
async def save_easy_chunk(tex_id: int | str, index: int, rewritten_text: str) -> None:
    paper_id = str(tex_id)
    key = f"{paper_id}:{index}"
    async with DB.session() as s:
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
    async with DB.session() as s:
        chunk = await s.get(Chunk, key)
        if not chunk:
            chunk = Chunk(key=key, paper_id=paper_id, index=index, image_path=image_path)
            s.add(chunk)
        else:
            chunk.image_path = image_path
        await s.commit()


async def save_math_result(
    tex_id: int,
    result_path: Optional[str] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> int:
    async with DB.session() as s:
        math = MathFile(tex_id=tex_id, file_addr=result_path)
        s.add(math)
        await s.commit()
        await s.refresh(math)
        return math.math_id


# ---- 결과 조회 & 조립 ----
async def fetch_results(tex_id: int | str) -> Optional[Dict[str, Any]]:
    paper_id = str(tex_id)
    async with DB.session() as s:
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
            {"math_id": m.math_id, "file_addr": m.file_addr}
            for m in math_rows.scalars().all()
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
        }


async def assemble_final(tex_id: int) -> Dict[str, Any]:
    data = await fetch_results(tex_id)
    return data if data else {"paper_id": tex_id, "items": [], "math": {"files": []}}


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
