from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app import crud, schemas, auth

router = APIRouter()

@router.post("/register")
async def register(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await crud.get_user_by_username(db, user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = await crud.create_user(db, user.username, user.password)
    return {"message": "User registered"}

@router.post("/login", response_model=schemas.Token)
async def login(user: schemas.UserLogin, db: AsyncSession = Depends(get_db)):
    db_user = await crud.get_user_by_username(db, user.username)
    if not db_user or not auth.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_access_token({"sub": db_user.username})
    return {"access_token": token, "token_type": "bearer"}
