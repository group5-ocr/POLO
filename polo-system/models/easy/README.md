# POLO Easy Model

논문 텍스트를 **일반인도 이해할 수 있는 쉬운 한국어**로 재해석하는 모델입니다.  
추가로 Viz 모델을 호출하여 이미지까지 생성하는 파이프라인을 제공합니다.

---

## 📦 Requirements
- Python 3.11+
- GPU (권장: NVIDIA RTX 시리즈)
- 설치 패키지: `requirements.easy.txt`

```bash
cd models/easy
python -m venv venv
venv\Scripts\activate      # (Windows)
pip install --upgrade pip
pip install -r requirements.easy.txt

cd models/easy
venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 5003
