from fastapi import FastAPI
from app.routes import user_routes, file_routes
from app.database import engine, Base

app = FastAPI()

# 라우터 연결
app.include_router(user_routes.router, prefix="/users", tags=["Users"])
app.include_router(file_routes.router, prefix="/files", tags=["Files"])

# DB 테이블 생성
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
async def root():
    return {"message": "POLO API is running"}