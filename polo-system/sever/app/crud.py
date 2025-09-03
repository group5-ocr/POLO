from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app import models
from app.auth import hash_password

async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(select(models.User).filter(models.User.username == username))
    return result.scalars().first()

async def create_user(db: AsyncSession, username: str, password: str):
    hashed_pw = hash_password(password)
    new_user = models.User(username=username, hashed_password=hashed_pw)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def create_paper(db: AsyncSession, filename: str, result_path: str, user_id: int):
    paper = models.Paper(
        filename=filename,
        result_path=result_path,
        owner_id=user_id
    )
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    return paper

async def get_user_files(db: AsyncSession, user_id: int):
    result = await db.execute(select(models.Paper).filter(models.Paper.owner_id == user_id))
    return result.scalars().all()

async def get_file_by_id(db: AsyncSession, file_id: int):
    result = await db.execute(select(models.Paper).filter(models.Paper.id == file_id))
    return result.scalars().first()
