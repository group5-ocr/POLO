#!/usr/bin/env python3
"""
YOLO 학습 자동 재개 스크립트
기존 체크포인트를 자동으로 찾아서 이어서 학습
"""
import os
import re
import subprocess
from pathlib import Path

def find_latest_checkpoint():
    """가장 최신 체크포인트 찾기"""
    output_dir = "outputs/yolo-easy-qlora"
    
    if not os.path.exists(output_dir):
        print("❌ YOLO 출력 디렉토리가 없습니다. 처음부터 학습합니다.")
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
            checkpoints.append(item)
    
    if not checkpoints:
        print("❌ 체크포인트가 없습니다. 처음부터 학습합니다.")
        return None
    
    # 체크포인트 번호로 정렬
    def extract_number(name):
        match = re.search(r'checkpoint-(\d+)', name)
        return int(match.group(1)) if match else 0
    
    latest = max(checkpoints, key=extract_number)
    checkpoint_path = os.path.join(output_dir, latest)
    
    print(f"✅ 최신 체크포인트 발견: {latest}")
    return checkpoint_path

def update_docker_compose_with_checkpoint(checkpoint_path):
    """Docker Compose 파일에 체크포인트 경로 추가"""
    compose_file = "docker-compose.yml"
    
    if not os.path.exists(compose_file):
        print(f"❌ {compose_file} 파일을 찾을 수 없습니다.")
        return False
    
    # 파일 읽기
    with open(compose_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 체크포인트 라인 찾기
    checkpoint_line = f"      --resume_from_checkpoint {checkpoint_path}"
    
    # 이미 체크포인트 라인이 있는지 확인
    if "--resume_from_checkpoint" in content:
        # 기존 체크포인트 라인 교체 (간단한 문자열 교체)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '--resume_from_checkpoint' in line:
                lines[i] = checkpoint_line
                break
        content = '\n'.join(lines)
        print(f"✅ 기존 체크포인트 경로를 업데이트했습니다.")
    else:
        # 체크포인트 라인 추가
        content = content.replace(
            "      --target_modules q_proj,k_proj,v_proj,o_proj",
            f"      --target_modules q_proj,k_proj,v_proj,o_proj\n{checkpoint_line}"
        )
        print(f"✅ 체크포인트 경로를 추가했습니다.")
    
    # 파일 쓰기
    with open(compose_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {compose_file} 업데이트 완료")
    return True

def run_training():
    """학습 실행"""
    print("\n🚀 YOLO 학습 시작...")
    
    # 기존 컨테이너 정리
    print("🧹 기존 컨테이너 정리 중...")
    subprocess.run(["docker-compose", "down"], capture_output=True)
    
    # 학습 실행
    print("🎯 YOLO 학습 실행 중...")
    print("📊 학습 진행 상황을 모니터링하려면:")
    print("   docker logs -f yolo-train")
    print("\n⏹️  학습을 중단하려면 Ctrl+C를 누르세요")
    print("=" * 50)
    
    try:
        subprocess.run(["docker-compose", "up", "yolo-train"], check=True)
        print("\n✅ YOLO 학습 완료!")
        return True
    except KeyboardInterrupt:
        print("\n⏹️  학습이 중단되었습니다.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 실패: {e}")
        return False

def main():
    print("🔄 YOLO 학습 자동 재개")
    print("=" * 50)
    
    # 1. 최신 체크포인트 찾기
    checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path:
        # 2. Docker Compose 업데이트
        print(f"📝 Docker Compose 파일 업데이트 중...")
        update_docker_compose_with_checkpoint(checkpoint_path)
        
        # 3. 학습 실행
        run_training()
    else:
        # 체크포인트가 없으면 처음부터 학습
        print("🆕 처음부터 학습을 시작합니다...")
        run_training()

if __name__ == "__main__":
    main()
