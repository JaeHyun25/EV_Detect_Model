import logging
import os
from datetime import datetime

def setup_logging(log_dir: str = "logs"):
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)    
    
    # 로그 파일명 설정 (날짜 포함)
    log_file = os.path.join(log_dir, f"ev_prediction_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,    # 로깅레벨 INFO이상 기록 - 일반적인 정보, 시스템의 정상적인 동작 기록 이상 동작시 기록
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__) 
