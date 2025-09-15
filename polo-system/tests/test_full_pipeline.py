#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 파이프라인 테스트 스크립트

논문 업로드 → arXiv 다운로드 → 전처리 → Math/Easy 모델 → Viz 연동까지의 전체 플로우를 테스트합니다.
"""

import asyncio
import httpx
import json
from pathlib import Path

# 테스트 설정
SERVER_URL = "http://localhost:8000"
ARXIV_ID = "1706.03762"  # Transformer 논문
USER_ID = 1

async def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("🚀 전체 파이프라인 테스트 시작")
    
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            # 1. arXiv 논문 업로드
            print(f"\n1️⃣ arXiv 논문 업로드: {ARXIV_ID}")
            response = await client.post(
                f"{SERVER_URL}/api/upload/from-arxiv",
                json={
                    "user_id": USER_ID,
                    "arxiv_id": ARXIV_ID,
                    "title": "Attention Is All You Need"
                }
            )
            
            if response.status_code != 200:
                print(f"❌ arXiv 업로드 실패: {response.status_code} - {response.text}")
                return False
                
            upload_result = response.json()
            print(f"✅ arXiv 업로드 성공: {upload_result}")
            
            # 2. 처리 상태 확인 (여러 번 체크)
            paper_id = upload_result.get("tex_id")
            if not paper_id:
                print("❌ paper_id를 찾을 수 없습니다")
                return False
                
            print(f"\n2️⃣ 처리 상태 확인: paper_id={paper_id}")
            
            for i in range(10):  # 최대 10번 체크
                await asyncio.sleep(10)  # 10초 대기
                
                status_response = await client.get(f"{SERVER_URL}/api/results/{paper_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"   체크 {i+1}: {status}")
                    
                    # 완료 확인
                    if status.get("status") == "completed":
                        print("✅ 전체 파이프라인 완료!")
                        return True
                else:
                    print(f"   체크 {i+1}: 상태 확인 실패 ({status_response.status_code})")
            
            print("⚠️ 타임아웃: 파이프라인이 완료되지 않았습니다")
            return False
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            return False

async def test_services_health():
    """서비스 상태 확인"""
    print("🔍 서비스 상태 확인")
    
    services = [
        ("서버", "http://localhost:8000/health"),
        ("전처리", "http://localhost:5002/health"),
        ("Easy 모델", "http://localhost:5003/health"),
        ("Math 모델", "http://localhost:5004/health"),
        ("Viz 모델", "http://localhost:5005/health"),
    ]
    
    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in services:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"✅ {name}: 정상")
                else:
                    print(f"❌ {name}: 오류 ({response.status_code})")
            except Exception as e:
                print(f"❌ {name}: 연결 실패 ({e})")

if __name__ == "__main__":
    print("=" * 60)
    print("POLO 전체 파이프라인 테스트")
    print("=" * 60)
    
    # 서비스 상태 확인
    asyncio.run(test_services_health())
    
    print("\n" + "=" * 60)
    
    # 전체 파이프라인 테스트
    success = asyncio.run(test_full_pipeline())
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 테스트 성공!")
    else:
        print("💥 테스트 실패!")
    print("=" * 60)
