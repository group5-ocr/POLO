import bcrypt

def hash_password(password: str) -> str:
    """
    비밀번호를 bcrypt로 해시화합니다.
    
    Args:
        password: 원본 비밀번호
        
    Returns:
        str: 해시화된 비밀번호
    """
    # 비밀번호를 바이트로 인코딩
    password_bytes = password.encode('utf-8')
    
    # bcrypt로 해시화 (salt 자동 생성)
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    
    # 바이트를 문자열로 디코딩하여 반환
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """
    비밀번호와 해시화된 비밀번호를 비교합니다.
    
    Args:
        password: 입력된 비밀번호
        hashed_password: 저장된 해시화된 비밀번호
        
    Returns:
        bool: 비밀번호가 일치하면 True, 아니면 False
    """
    try:
        # 비밀번호를 바이트로 인코딩
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        
        # bcrypt로 검증
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        print(f"비밀번호 검증 오류: {e}")
        return False
