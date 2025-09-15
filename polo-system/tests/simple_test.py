#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import httpx

async def test_server():
    """간단한 서버 테스트"""
    
    print("🔍 서버 연결 테스트...")
    
    try:
        async with httpx.AsyncClient() as client:
            # 1. 모델 상태 확인
            print("\n1️⃣ 모델 상태 확인...")
            response = await client.get("http://localhost:8000/api/model-status", timeout=5.0)
            print(f"   상태 코드: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   응답: {data}")
            else:
                print(f"   오류: {response.text}")
            
            # 2. arXiv 업로드 테스트
            print("\n2️⃣ arXiv 업로드 테스트...")
            response = await client.post(
                "http://localhost:8000/api/from-arxiv",
                json={
                    "user_id": 1,
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need"
                },
                timeout=30.0
            )
            print(f"   상태 코드: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   응답: {data}")
            else:
                print(f"   오류: {response.text}")
                
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    asyncio.run(test_server())
