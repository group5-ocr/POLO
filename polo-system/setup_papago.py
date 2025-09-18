#!/usr/bin/env python3
"""
Papago API 설정 스크립트
"""
import os
from pathlib import Path

def setup_papago_env():
    """Papago API 환경 변수 설정"""
    
    print("=== Papago API 설정 ===")
    print("네이버 개발자 센터에서 Papago API 키를 발급받아주세요:")
    print("https://developers.naver.com/apps/#/myapps")
    print()
    
    client_id = input("Papago Client ID를 입력하세요: ").strip()
    client_secret = input("Papago Client Secret을 입력하세요: ").strip()
    
    if not client_id or not client_secret:
        print("❌ API 키가 입력되지 않았습니다.")
        return False
    
    # 환경 변수 설정
    os.environ["PAPAGO_CLIENT_ID"] = client_id
    os.environ["PAPAGO_CLIENT_SECRET"] = client_secret
    
    # 번역 방법 선택
    print("\n번역 방법을 선택하세요:")
    print("1. 단일 번역 (영어 → 한국어)")
    print("2. 체인 번역 (영어 → 일본어 → 한국어)")
    
    choice = input("선택 (1 또는 2): ").strip()
    if choice == "2":
        os.environ["PAPAGO_USE_CHAIN"] = "true"
        print("✅ 체인 번역이 설정되었습니다.")
    else:
        os.environ["PAPAGO_USE_CHAIN"] = "false"
        print("✅ 단일 번역이 설정되었습니다.")
    
    print("\n✅ Papago API 설정이 완료되었습니다!")
    print("이제 Easy 모델을 실행하면 Papago 번역을 사용합니다.")
    
    return True

def test_papago():
    """Papago API 테스트"""
    try:
        import requests
        
        client_id = os.getenv("PAPAGO_CLIENT_ID")
        client_secret = os.getenv("PAPAGO_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            print("❌ API 키가 설정되지 않았습니다.")
            return False
        
        url = "https://openapi.naver.com/v1/papago/n2mt"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        data = {
            "source": "en",
            "target": "ko",
            "text": "Hello, this is a test."
        }
        
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            result = response.json()
            translated = result.get("message", {}).get("result", {}).get("translatedText", "")
            print(f"✅ 테스트 성공: 'Hello, this is a test.' → '{translated}'")
            return True
        else:
            print(f"❌ API 테스트 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    if setup_papago_env():
        print("\n=== API 테스트 ===")
        test_papago()
