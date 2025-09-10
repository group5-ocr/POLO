
## 도커

### 이미지 빌드 및 학습 실행 코드

1. cd polo-system/models/fine-tuning

2. 빌드 생성 (Dockerfile, requirements.train.txt 변경)
docker compose build --no-cache easy-train

3. qlora.py, train.jsonl, 파라미터 수정 시에
docker compose up easy-train

------

### 이미지 및 컨테이너 관리 (powershell)

1. docker 빌드 캐시 삭제
docker builder prune -af

2. 모든 컨테이너 삭제
docker ps -aq | ForEach-Object { docker rm -f $_ }

3. 사용하지 않는 볼륨 삭제
docker volume ls -q | ForEach-Object { docker volume rm $_ }

4. 모든 이미지 삭제 (powershell)
docker images -aq | ForEach-Object { docker rmi -f $_ }

5. 초기화
docker system prune -a --volumes -f
