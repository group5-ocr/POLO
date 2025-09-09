# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 모듈 alias로 불러 충돌 방지
from routes import convert, upload, results, generate as easy_generate

def create_app() -> FastAPI:
    app = FastAPI(title="POLO Easy Inference API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 필요 시 프론트 도메인만 허용으로 좁혀도 됨
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ✅ easy 전용 prefix
    app.include_router(upload.router,        prefix="/easy")
    app.include_router(convert.router,       prefix="/easy")
    app.include_router(results.router,       prefix="/easy")
    app.include_router(easy_generate.router, prefix="/easy")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app

app = create_app()