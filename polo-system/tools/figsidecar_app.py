# -*- coding: utf-8 -*-
"""
Figure 사이드카 정적 서버 (선택적)
메인 서버에 /static 마운트가 없을 때 별도 포트로 정적 파일만 서빙
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import argparse


def main():
    parser = argparse.ArgumentParser(description="Figure Sidecar Static Server")
    parser.add_argument("--root", 
                       default=r"C:\POLO\POLO\polo-system\server\data\outputs",
                       help="Static files root directory")
    parser.add_argument("--port", type=int, default=8020,
                       help="Server port (default: 8020)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Server host (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    # 정적 파일 루트 확인
    static_root = Path(args.root)
    if not static_root.exists():
        print(f"⚠️ 정적 파일 루트 없음: {static_root}")
        print("   디렉터리를 생성합니다...")
        static_root.mkdir(parents=True, exist_ok=True)
    
    # FastAPI 앱 생성
    app = FastAPI(
        title="FigSidecar Static Server",
        description="Figure 사이드카 정적 파일 서버",
        version="1.0.0"
    )
    
    # 정적 파일 마운트
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    
    # 헬스 체크 엔드포인트
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "static_root": str(static_root),
            "port": args.port
        }
    
    print(f"🚀 Figure Sidecar Server 시작")
    print(f"   포트:        {args.port}")
    print(f"   정적 루트:   {static_root}")
    print(f"   URL:         http://localhost:{args.port}/static/")
    print(f"   헬스 체크:   http://localhost:{args.port}/health")
    
    # 서버 실행
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
