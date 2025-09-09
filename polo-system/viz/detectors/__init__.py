# 패키지 인식용 파일
# python viz/render.py처럼 파일을 직접 실행하면 이때 viz는 “패키지 컨텍스트” 없이 동작하고, 내부에서 import가 패키지 기반 경로를 기대하면 문제가 생길 수 있음.

# pkgutil.iter_modules + importlib.import_module("viz.templates.grammars.xxx") 같은 점(.)으로 구분된 모듈 임포트는, 상위 폴더들이 패키지로 인식돼야 안정적으로 동작합니다. 즉, viz/, viz/templates/, viz/templates/grammars/에 __init__.py가 있으면 안전.