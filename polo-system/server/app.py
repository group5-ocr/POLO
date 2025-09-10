from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import convert, results  # upload는 안 쓰면 생략

app = FastAPI(title="POLO Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(convert.router, prefix="/easy")
app.include_router(results.router, prefix="/easy")

@app.get("/health")
def health():
    return {"status": "ok"}
