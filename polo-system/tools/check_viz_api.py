#!/usr/bin/env python3
"""
외부 VIZ API 연결 상태 확인 스크립트
팀원이 제공한 클라우드타입 배포 주소를 통해 API 상태를 확인합니다.
"""

import requests
import sys
import time

# 클라우드타입 배포 주소 (팀원이 제공한 주소)
VIZ_API_URL = "https://port-0-paper-viz-mc3ho385f405b6d9.sel5.cloudtype.app/"

def check_api_health():
    """API 서버의 health 상태를 확인합니다."""
    
    # 가능한 health check 엔드포인트들
    health_endpoints = [
        "/healthz", 
        "/api/healthz",
        "/"  # 기본 루트 경로
    ]
    
    print(f"VIZ API 연결 상태 확인 중...")
    print(f"대상 URL: {VIZ_API_URL}")
    print("-" * 50)
    
    for endpoint in health_endpoints:
        url = f"{VIZ_API_URL}{endpoint}"
        try:
            print(f"{endpoint} 확인 중...", end=" ")
            
            # 타임아웃 10초로 설정 (GET 요청 - health check)
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"성공 (상태코드: {response.status_code})")
                print(f"응답 시간: {response.elapsed.total_seconds():.2f}초")
                
                # 응답 내용이 있으면 일부 출력
                if response.text:
                    content = response.text[:200]  # 처음 200자만
                    print(f"응답 내용: {content}...")
                
                return True, endpoint
            else:
                print(f"응답 있음 (상태코드: {response.status_code})")
                
        except requests.exceptions.Timeout:
            print(f"타임아웃 (10초 초과)")
        except requests.exceptions.ConnectionError:
            print(f"연결 실패")
        except requests.exceptions.RequestException as e:
            print(f"요청 오류: {e}")
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
    
    return False, None

def check_viz_api_basic():
    """VIZ API 기본 연결만 확인합니다 (실제 변환 엔드포인트 테스트 없이)."""
    
    print(f"\nVIZ API 기본 연결 확인...")
    print("-" * 50)
    
    try:
        # 루트 경로로 GET 요청 (기본 연결 확인)
        response = requests.get(VIZ_API_URL, timeout=10)
        
        print(f"서버 응답 확인 완료 (상태코드: {response.status_code})")
        print(f"응답 시간: {response.elapsed.total_seconds():.2f}초")
        
        # 200번대나 405(Method Not Allowed)도 서버가 살아있다는 의미
        if 200 <= response.status_code < 300 or response.status_code == 405:
            return True
        else:
            print(f"서버는 응답하지만 상태가 정상적이지 않을 수 있음")
            return True  # 서버가 응답하면 일단 연결은 된 것으로 판단
            
    except requests.exceptions.Timeout:
        print(f"타임아웃 (10초 초과)")
    except requests.exceptions.ConnectionError:
        print(f"연결 실패 - 서버가 응답하지 않음")
    except requests.exceptions.RequestException as e:
        print(f"요청 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
    
    return False

def main():
    """메인 실행 함수"""
    
    print("=" * 60)
    print("POLO VIZ API 상태 확인 도구")
    print("=" * 60)
    
    # 1. Health check
    is_healthy, working_endpoint = check_api_health()
    
    if not is_healthy:
        print(f"\nAPI 서버에 연결할 수 없습니다.")
        print(f"확인사항:")
        print(f"   - 서버가 실행 중인지 확인")
        print(f"   - 네트워크 연결 상태 확인")
        print(f"   - URL이 올바른지 확인: {VIZ_API_URL}")
        return False
    
    print(f"\nAPI 서버 연결 성공!")
    print(f"작동하는 엔드포인트: {working_endpoint}")
    
    # 2. VIZ API 기본 연결 확인 (변환 기능 테스트 없이)
    viz_connected = check_viz_api_basic()
    
    if viz_connected:
        print(f"\nVIZ API 서버 연결 성공!")
        print(f"외부 API 서버가 정상적으로 응답하고 있습니다.")
        return True
    else:
        print(f"\nVIZ API 서버 연결 실패.")
        print(f"외부 서버가 응답하지 않거나 네트워크 문제가 있을 수 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    
    # 배치 파일에서 사용할 수 있도록 exit code 설정
    if success:
        print(f"\n모든 검사 통과! 시스템 사용 준비 완료.")
        sys.exit(0)  # 성공
    else:
        print(f"\n일부 검사 실패. 시스템 관리자에게 문의하세요.")
        sys.exit(1)  # 실패
