# YOLOv1 논문 통합 분석 설정 파일

class Config:
    """설정 클래스"""
    
    # 데이터 파일 경로
    EASY_JSONL_PATH = "data/easy_model_result.jsonl"
    MATH_JSONL_PATH = "data/math_model_result.jsonl"
    VIZ_JSONL_PATH = "data/viz_model_result.jsonl"
    INTEGRATED_JSONL_PATH = "data/integrated_result.jsonl"
    
    # 논문 기본 정보
    PAPER_INFO = {
        "paper_id": "yolov1_paper",
        "paper_title": "You Only Look Once: Unified, Real-Time Object Detection",
        "paper_authors": "Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi",
        "paper_date": "2016-06-27",
        "paper_venue": "CVPR 2016",
        "paper_doi": "10.1109/CVPR.2016.350"
    }
    
    # YOLOv1 논문 구조 정보 (라인 번호 기준)
    PAPER_STRUCTURE = {
        "abstract": {"start": 1, "end": 50, "title": "초록"},
        "introduction": {"start": 51, "end": 200, "title": "서론"},
        "methods": {"start": 201, "end": 400, "title": "방법론"},
        "results": {"start": 401, "end": 600, "title": "실험 결과"},
        "discussion": {"start": 601, "end": 800, "title": "토론"},
        "conclusion": {"start": 801, "end": 900, "title": "결론"}
    }
    
    # 섹션 순서
    SECTION_ORDER = [
        "abstract", "introduction", "methods", 
        "results", "discussion", "conclusion"
    ]
    
    # 데이터 소스 모드 (mockup 또는 real)
    DATA_SOURCE_MODE = "mockup"
    
    @classmethod
    def get_section_order(cls, section_name):
        """섹션 순서 반환"""
        try:
            return cls.SECTION_ORDER.index(section_name) + 1
        except ValueError:
            return 0
    
    @classmethod
    def get_section_title(cls, section_name):
        """섹션 제목 반환"""
        return cls.PAPER_STRUCTURE.get(section_name, {}).get("title", section_name)
