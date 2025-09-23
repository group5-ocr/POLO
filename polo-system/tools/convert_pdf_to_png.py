#!/usr/bin/env python3
"""
PDF를 PNG로 변환하는 스크립트
팀원이 제공한 외부 VIZ API를 사용하여 arxiv 논문을 PNG 슬라이드로 변환합니다.
포트 8010을 사용합니다.
"""

import requests
import zipfile
import io
import os
import sys
import argparse
from pathlib import Path

# 클라우드타입 배포 주소 (팀원 제공, 포트 8010 사용)
VIZ_API_URL = "https://port-0-paper-viz-mc3ho385f405b6d9.sel5.cloudtype.app:8010"

def convert_arxiv_to_png(arxiv_id, output_dir=None):
    """
    ArXiv ID를 받아서 PNG 슬라이드로 변환합니다.
    
    Args:
        arxiv_id (str): ArXiv 논문 ID (예: "1506.02640")
        output_dir (str): 출력 디렉토리 (기본: ./slides_{arxiv_id})
    
    Returns:
        bool: 성공 여부
        str: 출력 디렉토리 경로
    """
    
    if not output_dir:
        output_dir = f"./slides_{arxiv_id}"
    
    # 절대 경로로 변환
    output_dir = os.path.abspath(output_dir)
    
    print(f"ArXiv ID: {arxiv_id}")
    print(f"VIZ API URL: {VIZ_API_URL}")
    print(f"출력 디렉토리: {output_dir}")
    print("-" * 50)
    
    # API 엔드포인트 URL 구성
    url = f"{VIZ_API_URL}/api/viz-api/generate-zip/{arxiv_id}"
    
    try:
        print(f"PNG 변환 요청 전송 중...")
        print(f"요청 URL: {url}")
        
        # POST 요청으로 ZIP 파일 요청
        response = requests.post(url, timeout=120)  # 2분 타임아웃
        
        # HTTP 에러 체크
        response.raise_for_status()
        
        print(f"변환 요청 성공! (상태코드: {response.status_code})")
        print(f"응답 크기: {len(response.content) / (1024*1024):.2f}MB")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        print(f"출력 디렉토리 생성: {output_dir}")
        
        # ZIP 파일 압축 해제
        print(f"ZIP 파일 압축 해제 중...")
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # ZIP 내용 확인
            file_list = z.namelist()
            png_files = [f for f in file_list if f.lower().endswith('.png')]
            
            print(f"ZIP 파일 내용:")
            print(f"   - 전체 파일 수: {len(file_list)}")
            print(f"   - PNG 파일 수: {len(png_files)}")
            
            if png_files:
                print(f"PNG 파일 목록:")
                for png_file in png_files[:10]:  # 처음 10개만 출력
                    print(f"   - {png_file}")
                if len(png_files) > 10:
                    print(f"   ... 그외 {len(png_files) - 10}개 파일")
            
            # 모든 파일 압축 해제
            z.extractall(output_dir)
        
        print(f"변환 완료!")
        print(f"PNG 파일들이 {output_dir} 폴더에 저장되었습니다.")
        
        return True, output_dir
        
    except requests.exceptions.Timeout:
        print(f"타임아웃: 변환 요청이 2분을 초과했습니다.")
        print(f"큰 논문의 경우 시간이 더 오래 걸릴 수 있습니다.")
        return False, None
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 오류: {e}")
        print(f"상태코드: {response.status_code}")
        if hasattr(response, 'text'):
            print(f"응답 내용: {response.text[:500]}")
        return False, None
        
    except requests.exceptions.ConnectionError:
        print(f"연결 오류: API 서버에 연결할 수 없습니다.")
        print(f"확인사항:")
        print(f"   - 네트워크 연결 상태")
        print(f"   - API 서버 상태 (check_viz_api.py 실행)")
        print(f"   - URL이 올바른지: {VIZ_API_URL}")
        return False, None
        
    except zipfile.BadZipFile:
        print(f"ZIP 파일 오류: 응답이 올바른 ZIP 파일이 아닙니다.")
        print(f"API 서버에서 올바른 형식의 응답을 보내지 않았을 수 있습니다.")
        return False, None
        
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        print(f"시스템 관리자에게 문의하세요.")
        return False, None

def validate_arxiv_id(arxiv_id):
    """ArXiv ID 형식을 검증합니다."""
    
    # 기본적인 ArXiv ID 패턴 검사
    if not arxiv_id:
        return False, "ArXiv ID가 비어있습니다."
    
    # 공백 제거
    arxiv_id = arxiv_id.strip()
    
    # 기본 길이 체크
    if len(arxiv_id) < 7:
        return False, f"ArXiv ID가 너무 짧습니다: {arxiv_id}"
    
    # 점(.)이 포함되어 있는지 확인
    if '.' not in arxiv_id:
        return False, f"ArXiv ID에 점(.)이 포함되어야 합니다: {arxiv_id}"
    
    return True, arxiv_id

def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(
        description="ArXiv 논문을 PNG 슬라이드로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python convert_pdf_to_png.py 1506.02640
  python convert_pdf_to_png.py 1506.02640 --output ./my_slides
  python convert_pdf_to_png.py 2103.00020 --output C:\\temp\\slides

지원하는 ArXiv ID 형식:
  - 1506.02640 (구 형식)
  - 2103.00020 (신 형식)
        """
    )
    
    parser.add_argument(
        'arxiv_id',
        help='변환할 ArXiv 논문 ID (예: 1506.02640)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='출력 디렉토리 경로 (기본: ./slides_{arxiv_id})',
        default=None
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='실제 변환 없이 API 연결만 확인'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POLO PDF → PNG 변환 도구")
    print("=" * 60)
    
    # ArXiv ID 검증
    is_valid, validated_id = validate_arxiv_id(args.arxiv_id)
    if not is_valid:
        print(f"ArXiv ID 오류: {validated_id}")
        return False
    
    # API 연결만 확인하는 경우
    if args.check_only:
        print(f"API 연결 상태만 확인합니다...")
        try:
            response = requests.head(f"{VIZ_API_URL}/api/viz-api/generate-zip/{validated_id}", timeout=10)
            if response.status_code in [200, 405]:  # 405는 HEAD 메서드를 지원하지 않는 경우
                print(f"API 연결 성공!")
                return True
            else:
                print(f"API 응답 이상 (상태코드: {response.status_code})")
                return False
        except Exception as e:
            print(f"API 연결 실패: {e}")
            return False
    
    # 실제 변환 수행
    success, output_path = convert_arxiv_to_png(validated_id, args.output)
    
    if success:
        print(f"\n변환 성공!")
        print(f"출력 위치: {output_path}")
        print(f"PNG 파일들을 확인해보세요!")
        return True
    else:
        print(f"\n변환 실패!")
        print(f"오류 메시지를 확인하고 다시 시도해주세요.")
        return False

if __name__ == "__main__":
    success = main()
    
    # 배치 파일에서 사용할 수 있도록 exit code 설정
    sys.exit(0 if success else 1)
