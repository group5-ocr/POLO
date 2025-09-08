import argparse
from texprep.pipeline import run_pipeline
from texprep.utils.cfg import load_cfg

def main():
    # 명령줄 인터페이스(CLI) 엔트리포인트
    # python -m texprep.cli --main ./paper.tex --to json
    # 이런 식으로 실행되면 여기서 시작한다.

    # argparse: 커맨드라인 인자를 파싱하는 표준 라이브러리
    ap = argparse.ArgumentParser()

    # --config : 어떤 설정 파일(YAML)을 쓸지 지정 (기본값은 configs/default.yaml)
    ap.add_argument("--config", default="configs/default.yaml")

    # --main : 필수 인자, 분석할 main.tex 경로
    ap.add_argument("--main", required=True)

    # --to : 산출물을 어디로 보낼지 (json 파일, postgres DB, 둘 다)
    ap.add_argument("--to", choices=["json","pg","both"], default="json")

    args = ap.parse_args()  # 실제 명령줄 인자 해석

    # 설정 불러오기 (YAML+환경변수 머지)
    cfg = load_cfg(args.config)

    # 파이프라인 실행
    run_pipeline(cfg, args.main, sink=args.to)

# 이 파일을 직접 실행했을 때만 main()을 돌려라.
if __name__ == "__main__":
    main()
