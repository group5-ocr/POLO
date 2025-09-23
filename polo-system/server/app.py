# server/app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from services.database.db import DB

from routes import upload, generate, results, math_generate, database

# .env 파일 로드
load_dotenv()

app = FastAPI(title="POLO Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "db_mode": DB.mode}

@app.on_event("startup")
async def startup():
    await DB.init()
    print(f"현재 DB 모드: {DB.mode}")

@app.on_event("shutdown")
async def shutdown():
    close_fn = getattr(DB, "close", None)
    if callable(close_fn):
        await close_fn()

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(generate.router, prefix="/generate", tags=["callbacks"])
app.include_router(results.router, prefix="/results", tags=["results"])
app.include_router(math_generate.router, prefix="/math", tags=["math"])
app.include_router(database.router, prefix="/db", tags=["database"])

# API 엔드포인트 추가
app.include_router(upload.router, prefix="/api", tags=["api"])
app.include_router(generate.router, prefix="/api", tags=["api"])
app.include_router(results.router, prefix="/api/results", tags=["api"])  # /api/results/{paper_id}/...
app.include_router(math_generate.router, prefix="/api", tags=["api"])

# 정적 파일: outputs/* (시각화/중간 산출물 제공)
BASE_DIR = Path(__file__).resolve().parent  # polo-system/server
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# /outputs/<paper_id>/... 을 브라우저에서 바로 접근
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# /static/<path> → outputs/<path> 매핑 (프론트엔드 호환성)
app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")

# viz 파일 진단용 엔드포인트
@app.get("/api/results/{paper_id}/viz-list")
async def list_viz_files(paper_id: str):
    root = OUTPUTS_DIR / paper_id / "viz"
    if not root.exists():
        return {"ok": False, "reason": "viz dir missing", "dir": str(root)}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            files.append(str(Path(dirpath).joinpath(fn).relative_to(OUTPUTS_DIR)).replace("\\", "/"))
    return {"ok": True, "count": len(files), "files": files}

# Figure 처리 엔드포인트
@app.post("/api/build-figures")
async def build_and_attach_figures(
    paper_id: str,
    assets_path: str = None,
    integrated_path: str = None
):
    """
    PDF/이미지를 PNG로 렌더링하고 [Figure] 토큰에 첨부
    """
    try:
        import sys
        sys.path.append(str(BASE_DIR.parent))
        from viz.assets_mapper import build_figure_index, get_figure_web_paths
        from viz.integrate_figures import attach_figures, validate_figure_integration
        
        # 기본 경로 설정
        paper_dir = OUTPUTS_DIR / paper_id
        if not assets_path:
            assets_path = str(paper_dir / "source" / "assets.jsonl")
        if not integrated_path:
            integrated_path = str(paper_dir / "integrated_result.json")
        
        assets_file = Path(assets_path)
        integrated_file = Path(integrated_path)
        
        if not assets_file.exists():
            raise HTTPException(status_code=404, detail=f"assets.jsonl 없음: {assets_path}")
        if not integrated_file.exists():
            raise HTTPException(status_code=404, detail=f"integrated_result.json 없음: {integrated_path}")
        
        # 소스 디렉토리 추정
        source_dir = assets_file.parent
        
        # 1. Figure 인덱스 구축
        figures = build_figure_index(assets_file, source_dir, OUTPUTS_DIR)
        
        if not figures:
            return {"ok": False, "reason": "No figures found", "figures_count": 0}
        
        # 2. 웹 경로 추가
        figures_with_web = get_figure_web_paths(figures, "/static")
        
        # 3. [Figure] 토큰에 첨부
        output_file = integrated_file.with_name("integrated_result.with_figures.json")
        attach_figures(integrated_file, output_file, figures_with_web, "/static")
        
        # 4. 검증
        validation = validate_figure_integration(output_file)
        
        return {
            "ok": True,
            "paper_id": paper_id,
            "figures_count": len(figures),
            "integrated_path": str(output_file),
            "validation": validation
        }
        
    except Exception as e:
        import traceback
        print(f"❌ [BUILD-FIGURES] 실패: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Figure 처리 실패: {str(e)}")

@app.get("/api/results/{paper_id}/figures")
async def get_paper_figures(paper_id: str):
    """
    논문의 figure 목록 조회
    """
    try:
        import sys
        sys.path.append(str(BASE_DIR.parent))
        from viz.assets_mapper import build_figure_index, get_figure_web_paths
        
        paper_dir = OUTPUTS_DIR / paper_id
        assets_file = paper_dir / "source" / "assets.jsonl"
        
        if not assets_file.exists():
            return {"ok": False, "reason": "assets.jsonl 없음", "figures": []}
        
        source_dir = assets_file.parent
        figures = build_figure_index(assets_file, source_dir, OUTPUTS_DIR)
        figures_with_web = get_figure_web_paths(figures, "/static")
        
        return {
            "ok": True,
            "paper_id": paper_id,
            "figures_count": len(figures),
            "figures": figures_with_web
        }
        
    except Exception as e:
        print(f"❌ [GET-FIGURES] 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Figure 조회 실패: {str(e)}")