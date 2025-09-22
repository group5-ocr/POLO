#!/usr/bin/env python3
"""
체크포인트 확인 및 docker-compose.yml 자동 수정 스크립트
"""
import os
import sys
import glob
import re
from pathlib import Path

def find_latest_checkpoint(output_dir: str) -> str:
    """가장 최신 체크포인트 찾기"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # 체크포인트 번호로 정렬
    def extract_number(path):
        match = re.search(r'checkpoint-(\d+)', path)
        return int(match.group(1)) if match else 0
    
    latest = max(checkpoints, key=extract_number)
    return latest

def update_docker_compose(checkpoint_path: str = None):
    """docker-compose.yml 파일 업데이트"""
    compose_file = "docker-compose.yml"
    
    if not os.path.exists(compose_file):
        print(f"❌ {compose_file} 파일을 찾을 수 없습니다.")
        return False
    
    # 파일 읽기
    with open(compose_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 체크포인트 라인 찾기 및 수정
    if checkpoint_path:
        # 체크포인트가 있으면 해당 경로로 설정
        new_line = f"      --resume_from_checkpoint {checkpoint_path}"
        print(f"✅ 체크포인트 발견: {checkpoint_path}")
    else:
        # 체크포인트가 없으면 처음부터 학습
        new_line = "      # --resume_from_checkpoint outputs/yolo-easy-qlora/checkpoint-4000"
        print("ℹ️  체크포인트 없음 - 처음부터 학습")
    
    # 정규식으로 체크포인트 라인 교체
    pattern = r'(\s+--resume_from_checkpoint\s+[^\s]+)'
    replacement = new_line
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
    else:
        # 체크포인트 라인이 없으면 추가
        content = content.replace(
            "      --target_modules q_proj,k_proj,v_proj,o_proj",
            f"      --target_modules q_proj,k_proj,v_proj,o_proj\n{new_line}"
        )
    
    # 파일 쓰기
    with open(compose_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {compose_file} 업데이트 완료")
    return True

def main():
    print("🔍 YOLO 파인튜닝 체크포인트 확인 중...")
    
    output_dir = "outputs/yolo-easy-qlora"
    
    # 체크포인트 찾기
    checkpoint = find_latest_checkpoint(output_dir)
    
    if checkpoint:
        print(f"📁 발견된 체크포인트: {checkpoint}")
        
        # 체크포인트 정보 출력
        checkpoint_num = re.search(r'checkpoint-(\d+)', checkpoint)
        if checkpoint_num:
            step_num = checkpoint_num.group(1)
            print(f"📊 체크포인트 단계: {step_num}")
            
            # 체크포인트 크기 확인
            if os.path.exists(checkpoint):
                size_mb = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(checkpoint)
                             for filename in filenames) / (1024 * 1024)
                print(f"💾 체크포인트 크기: {size_mb:.1f} MB")
    else:
        print("🆕 체크포인트 없음 - 처음부터 학습")
    
    # docker-compose.yml 업데이트
    update_docker_compose(checkpoint)
    
    print("\n🚀 다음 명령어로 학습 시작:")
    print("docker-compose up yolo-train")

if __name__ == "__main__":
    main()
