#!/usr/bin/env python3
"""
Google Translate API 라이브러리 설치 스크립트
"""
import subprocess
import sys

def install_google_translate():
    """Google Cloud Translate 라이브러리 설치"""
    try:
        print("📦 Google Cloud Translate 라이브러리 설치 중...")
        
        # Google Cloud Translate 라이브러리 설치
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "google-cloud-translate==2.0.1"
        ])
        
        print("✅ Google Cloud Translate 라이브러리 설치 완료!")
        print("이제 GOOGLE_API_KEY 환경 변수를 설정하고 테스트할 수 있습니다.")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 설치 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    install_google_translate()
