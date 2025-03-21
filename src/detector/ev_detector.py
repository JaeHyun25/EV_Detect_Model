import cv2
import numpy as np
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import time
import json
from .ev_classifier import EVClassifier

@dataclass
class DetectionResult:
    """검출 결과 데이터 클래스"""
    plate_number: str
    is_ev: bool
    confidence: float
    timestamp: datetime
    processing_time: float
    plate_area: Dict  # 번호판 위치 정보

class EVDetector:
    def __init__(self, xgb_model_path: str, lgbm_model_path: str):
        """초기화"""
        try:
            self.classifier = EVClassifier(xgb_model_path, lgbm_model_path)
            self.logger = logging.getLogger(__name__)
            self.logger.info("EVDetector 초기화 완료")
        except Exception as e:
            self.logger.error(f"EVDetector 초기화 중 오류 발생: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray, plate_info: Dict) -> DetectionResult:
        """단일 프레임 처리"""
        try:
            start_time = time.time()
            
            # plate_info 구조 로깅
            self.logger.info(f"입력된 plate_info 구조: {json.dumps(plate_info, indent=2, ensure_ascii=False)}")
            
            # 번호판 정보 추출
            if isinstance(plate_info, list) and len(plate_info) > 0:
                plate_info = plate_info[0]
            
            # area 정보 추출
            area = plate_info.get('area', {})
            if not area:
                raise ValueError("번호판 영역 정보를 찾을 수 없습니다.")
            
            # 이미지 처리 및 예측
            is_ev, processing_time = self.classifier.process_frame(frame, plate_info)
            
            # 결과 생성
            return DetectionResult(
                plate_number=plate_info.get('text', ''),
                is_ev=is_ev,
                confidence=0.0,  # TODO: 신뢰도 계산 로직 추가
                timestamp=datetime.now(),
                processing_time=processing_time,
                plate_area=area
            )
            
        except Exception as e:
            self.logger.error(f"프레임 처리 중 오류 발생: {str(e)}")
            raise

    def process_batch(self, frame: np.ndarray, plate_infos: List[Dict]) -> List[DetectionResult]:
        """여러 번호판 일괄 처리"""
        results = []
        for plate_info in plate_infos:
            result = self.process_frame(frame, plate_info)
            results.append(result)
        return results

    def save_results(self, results: List[DetectionResult], output_path: str):
        """결과 저장"""
        try:
            # 결과를 딕셔너리 리스트로 변환
            results_dict = [
                {
                    'plate_number': r.plate_number,
                    'is_ev': r.is_ev,
                    'confidence': r.confidence,
                    'processing_time': r.processing_time,
                    'timestamp': r.timestamp.isoformat(),
                    'plate_area': r.plate_area
                }
                for r in results
            ]
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"결과가 {output_path}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {str(e)}")
            raise 