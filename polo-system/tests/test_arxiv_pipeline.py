#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
arXiv 파이프라인 테스트 스크립트
"""

import asyncio
import httpx
import json
from pathlib import Path
import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_arxiv_pipeline():
    """arXiv 파이프라인 전체 테스트"""
    
    # 테스트용 arXiv ID (Transformer 논문)
    arxiv_id = "1706.03762"
    title = "Attention Is All You Need"
    
    print(f"🚀 arXiv 파이프라인 테스트 시작: {arxiv_id}")
    print(f"📄 논문 제목: {title}")
    
    # 1. arXiv 업로드 테스트
    print("\n1️⃣ arXiv 업로드 테스트...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/from-arxiv",
                json={
                    "user_id": 1,
                    "arxiv_id": arxiv_id,
                    "title": title
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ arXiv 업로드 성공!")
                print(f"   - tex_id: {data.get('tex_id')}")
                print(f"   - arxiv_id: {data.get('arxiv_id')}")
                print(f"   - 상태: {data.get('status')}")
                tex_id = data.get('tex_id')
            else:
                print(f"❌ arXiv 업로드 실패: {response.status_code}")
                print(f"   응답: {response.text}")
                return
                
    except Exception as e:
        print(f"❌ arXiv 업로드 오류: {e}")
        return
    
    # 2. 모델 상태 확인
    print("\n2️⃣ 모델 상태 확인...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/model-status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ 모델 상태:")
                print(f"   - Easy 모델: {'🟢' if status.get('easy_model') else '🔴'}")
                print(f"   - Math 모델: {'🟢' if status.get('math_model') else '🔴'}")
                print(f"   - 전체 상태: {status.get('status')}")
            else:
                print(f"❌ 모델 상태 확인 실패: {response.status_code}")
    except Exception as e:
        print(f"❌ 모델 상태 확인 오류: {e}")
    
    # 3. 다운로드 정보 확인
    if tex_id:
        print(f"\n3️⃣ 다운로드 정보 확인 (tex_id: {tex_id})...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:8000/download/info/{tex_id}")
                if response.status_code == 200:
                    info = response.json()
                    print(f"✅ 다운로드 정보:")
                    for category, files in info.get('files', {}).items():
                        if files:
                            print(f"   - {category}: {len(files)}개 파일")
                            for file in files[:3]:  # 처음 3개만 표시
                                print(f"     * {file.get('name')} ({file.get('size', 0)} bytes)")
                        else:
                            print(f"   - {category}: 파일 없음")
                else:
                    print(f"❌ 다운로드 정보 조회 실패: {response.status_code}")
        except Exception as e:
            print(f"❌ 다운로드 정보 조회 오류: {e}")
    
    # 4. 실제 파일 확인
    print(f"\n4️⃣ 실제 파일 확인...")
    
    # arXiv 다운로드 폴더 확인
    arxiv_dir = Path("../data/arxiv")
    if arxiv_dir.exists():
        arxiv_files = list(arxiv_dir.rglob(f"*{arxiv_id}*"))
        print(f"📁 arXiv 폴더 파일들:")
        for file in arxiv_files[:5]:  # 처음 5개만 표시
            print(f"   - {file.name} ({file.stat().st_size} bytes)")
    
    # 출력 폴더 확인
    output_dir = Path("../data/outputs")
    if output_dir.exists():
        output_files = list(output_dir.rglob("*"))
        print(f"📁 출력 폴더 파일들:")
        for file in output_files[:5]:  # 처음 5개만 표시
            print(f"   - {file.name} ({file.stat().st_size} bytes)")
    
    print(f"\n🎉 파이프라인 테스트 완료!")

async def test_pdf_upload():
    """PDF 업로드 테스트"""
    
    print(f"\n📄 PDF 업로드 테스트...")
    
    # 간단한 테스트 PDF 생성 (실제로는 기존 PDF 사용)
    test_pdf_path = Path("../data/raw/test.pdf")
    if not test_pdf_path.exists():
        print(f"❌ 테스트 PDF 파일이 없습니다: {test_pdf_path}")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            with open(test_pdf_path, "rb") as f:
                files = {"file": ("test.pdf", f, "application/pdf")}
                response = await client.post(
                    "http://localhost:8000/api/convert",
                    files=files,
                    timeout=30.0
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ PDF 업로드 성공!")
                print(f"   - doc_id: {data.get('doc_id')}")
                print(f"   - 파일명: {data.get('filename')}")
                print(f"   - 파일 크기: {data.get('file_size')} bytes")
                print(f"   - 추출된 텍스트 길이: {data.get('extracted_text_length')}")
                print(f"   - arXiv ID: {data.get('arxiv_id', '없음')}")
            else:
                print(f"❌ PDF 업로드 실패: {response.status_code}")
                print(f"   응답: {response.text}")
                
    except Exception as e:
        print(f"❌ PDF 업로드 오류: {e}")

async def test_models_directly():
    """모델 직접 테스트"""
    
    print(f"\n🤖 모델 직접 테스트...")
    
    # Easy 모델 테스트
    print(f"\n📝 Easy 모델 테스트...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:5002/batch",
                json={
                    "chunks_jsonl_path": "../data/outputs/test_chunks.jsonl"
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Easy 모델 성공!")
                print(f"   - 처리된 청크 수: {data.get('processed_chunks', 0)}")
            else:
                print(f"❌ Easy 모델 실패: {response.status_code}")
                print(f"   응답: {response.text}")
                
    except Exception as e:
        print(f"❌ Easy 모델 오류: {e}")
    
    # Math 모델 테스트
    print(f"\n🧮 Math 모델 테스트...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:5003/math",
                json={
                    "tex_file_path": "../data/outputs/test_math.tex"
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Math 모델 성공!")
                print(f"   - 처리된 방정식 수: {data.get('equation_count', 0)}")
            else:
                print(f"❌ Math 모델 실패: {response.status_code}")
                print(f"   응답: {response.text}")
                
    except Exception as e:
        print(f"❌ Math 모델 오류: {e}")

if __name__ == "__main__":
    print("🧪 POLO 파이프라인 테스트 시작")
    print("=" * 50)
    
    # 서버 상태 확인
    print("🔍 서버 상태 확인...")
    try:
        import httpx
        import asyncio
        
        async def check_server():
            try:
                async with httpx.AsyncClient() as client:
                    # /health 대신 /api/model-status로 확인
                    response = await client.get("http://localhost:8000/api/model-status", timeout=5.0)
                    if response.status_code == 200:
                        print("✅ 메인 서버 실행 중")
                        return True
                    else:
                        print(f"❌ 메인 서버 응답 오류: {response.status_code}")
                        return False
            except Exception as e:
                print(f"❌ 메인 서버 연결 실패: {e}")
                return False
        
        server_running = asyncio.run(check_server())
        
        if server_running:
            # 전체 파이프라인 테스트
            asyncio.run(test_arxiv_pipeline())
            
            # PDF 업로드 테스트
            asyncio.run(test_pdf_upload())
            
            # 모델 직접 테스트
            asyncio.run(test_models_directly())
        else:
            print("❌ 서버가 실행되지 않았습니다. run_system.bat을 먼저 실행하세요.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
