#!/usr/bin/env python3
"""
YOLO 논문 파인튜닝 실행 스크립트
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """필수 요구사항 확인"""
    print("🔍 시스템 요구사항 확인 중...")
    
    # Docker 확인
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker가 설치되지 않았습니다.")
            return False
    except FileNotFoundError:
        print("❌ Docker가 설치되지 않았습니다.")
        return False
    
    # Docker Compose 확인
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose: {result.stdout.strip()}")
        else:
            print("❌ Docker Compose가 설치되지 않았습니다.")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose가 설치되지 않았습니다.")
        return False
    
    # GPU 확인
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 감지됨")
        else:
            print("⚠️  NVIDIA GPU를 찾을 수 없습니다. CPU로 실행됩니다.")
    except FileNotFoundError:
        print("⚠️  nvidia-smi를 찾을 수 없습니다. CPU로 실행됩니다.")
    
    return True

def check_data():
    """YOLO 데이터 확인"""
    print("\n📚 YOLO 데이터 확인 중...")
    
    yolo_file = "training/yolo_v1.json"
    if not os.path.exists(yolo_file):
        print(f"❌ YOLO 데이터 파일을 찾을 수 없습니다: {yolo_file}")
        return False
    
    # 파일 크기 확인
    size_mb = os.path.getsize(yolo_file) / (1024 * 1024)
    print(f"✅ YOLO 데이터: {yolo_file} ({size_mb:.1f} MB)")
    
    return True

def check_checkpoints():
    """체크포인트 확인"""
    print("\n💾 체크포인트 확인 중...")
    
    yolo_output = "outputs/yolo-easy-qlora"
    if os.path.exists(yolo_output):
        checkpoints = [d for d in os.listdir(yolo_output) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            print(f"✅ 기존 체크포인트 발견: {latest}")
            print(f"📁 체크포인트 경로: {yolo_output}/{latest}")
            return latest
        else:
            print("ℹ️  체크포인트 없음 - 처음부터 학습")
            return None
    else:
        print("ℹ️  출력 디렉토리 없음 - 처음부터 학습")
        return None

def run_training():
    """YOLO 학습 실행"""
    print("\n🚀 YOLO 논문 파인튜닝 시작...")
    print("=" * 50)
    
    # Docker Compose로 학습 실행
    cmd = ["docker-compose", "up", "yolo-train"]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("\n📊 학습 진행 상황을 모니터링하려면:")
    print("   docker logs -f yolo-train")
    print("\n⏹️  학습을 중단하려면 Ctrl+C를 누르세요")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ YOLO 학습 완료!")
        return True
    except KeyboardInterrupt:
        print("\n⏹️  학습이 중단되었습니다.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 실패: {e}")
        return False

def run_serving():
    """YOLO 서빙 실행"""
    print("\n🌐 YOLO 서빙 서버 시작...")
    
    cmd = ["docker-compose", "up", "yolo-llm"]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("\n📡 서버 접속:")
    print("   http://localhost:5004/docs (API 문서)")
    print("   http://localhost:5004/health (헬스 체크)")
    print("   http://localhost:5004/yolo_test (테스트)")
    print("\n⏹️  서버를 중단하려면 Ctrl+C를 누르세요")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  서버가 중단되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 서버 실행 실패: {e}")

def run_auto_resume():
    """자동 재개 학습 실행"""
    print("\n🔄 자동 재개 학습 시작...")
    print("=" * 50)
    
    # 체크포인트 확인
    checkpoint = check_checkpoints()
    
    if checkpoint:
        print(f"✅ 체크포인트에서 이어서 학습: {checkpoint}")
        print("📊 Early Stopping이 활성화되어 있습니다.")
        print("   - 3번의 evaluation에서 개선이 없으면 자동 중단")
        print("   - 100 step마다 validation 실행")
        print("\n⏹️  학습을 중단하려면 Ctrl+C를 누르세요")
        print("=" * 50)
        
        try:
            subprocess.run(["docker-compose", "up", "yolo-train"], check=True)
            print("\n✅ 자동 재개 학습 완료!")
            return True
        except KeyboardInterrupt:
            print("\n⏹️  학습이 중단되었습니다.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 학습 실패: {e}")
            return False
    else:
        print("❌ 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
        return run_training()

def main():
    print("🎯 YOLO 논문 Easy 모델 파인튜닝")
    print("=" * 50)
    
    # 1. 요구사항 확인
    if not check_requirements():
        print("\n❌ 요구사항을 충족하지 않습니다.")
        return
    
    # 2. 데이터 확인
    if not check_data():
        print("\n❌ YOLO 데이터를 찾을 수 없습니다.")
        return
    
    # 3. 체크포인트 확인
    checkpoint = check_checkpoints()
    
    # 4. 사용자 선택
    print("\n🎮 실행 옵션을 선택하세요:")
    print("1. YOLO 학습만 실행")
    print("2. YOLO 서빙만 실행")
    print("3. 학습 후 서빙 실행")
    print("4. 자동 재개 학습 (체크포인트에서 이어서)")
    print("5. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-5): ").strip()
            if choice == "1":
                run_training()
                break
            elif choice == "2":
                run_serving()
                break
            elif choice == "3":
                if run_training():
                    print("\n⏳ 학습 완료! 서빙을 시작합니다...")
                    time.sleep(2)
                    run_serving()
                break
            elif choice == "4":
                run_auto_resume()
                break
            elif choice == "5":
                print("👋 종료합니다.")
                break
            else:
                print("❌ 잘못된 선택입니다. 1-5 중에서 선택하세요.")
        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break

if __name__ == "__main__":
    main()
