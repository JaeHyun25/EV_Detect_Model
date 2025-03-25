import cv2
import numpy as np
import joblib
import logging
from typing import List, Tuple, Dict, Union, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from ..utils.image_processing import preprocess_image, extract_features, validate_plate_info

@dataclass
class ProcessingMetrics:    # 메트릭 추적을 위한 데이터클래스 추가가
    """처리 메트릭 데이터 클래스"""
    elapsed_time: float     # 처리 시간
    confidence_score: float # 신뢰도 점수
    model_used: str  # 'xgb' or 'lgb'   # 사용된 모델
    error_occurred: bool = False # 오류 발생 여부
    error_message: Optional[str] = None # 오류 메시지

class EVClassifier:
    def __init__(self, 
                 xgb_model_path: str, 
                 lgb_model_path: str,
                 confidence_threshold: float = 0.45,
                 max_processing_time: float = 1.0,  # 최대 처리 시간 제한
                 max_workers: int = 4,              # 병렬 처리 워커 수 설정
                 batch_size: int = 32):             # 배치 처리 크기 설정정
        """전기차 판별 모델 로드
        
        Args:
            xgb_model_path (str): XGBoost 모델 경로
            lgb_model_path (str): LightGBM 모델 경로
            confidence_threshold (float): 예측 신뢰도 임계값
            max_processing_time (float): 최대 처리 시간 (초)
            max_workers (int): 병렬 처리 최대 워커 수
            batch_size (int): 배치 처리 크기
        """
        try:
            self.xgb_model = joblib.load(xgb_model_path)
            self.lgb_model = joblib.load(lgb_model_path)
            self.confidence_threshold = confidence_threshold
            self.max_processing_time = max_processing_time
            self.max_workers = max_workers
            self.batch_size = batch_size
            self.logger = logging.getLogger(__name__)
            self.metrics_history: List[ProcessingMetrics] = []
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray, plate_info: Dict) -> Tuple[bool, ProcessingMetrics]:
        """단일 프레임 처리 및 예측
        
        Args:
            frame (np.ndarray): 입력 이미지
            plate_info (Dict): 번호판 정보
            
        Returns:
            Tuple[bool, ProcessingMetrics]: (예측 결과, 처리 메트릭)
        """
        try:
            start_time = time.time()
            
            # 입력 검증
            if not isinstance(frame, np.ndarray):
                raise TypeError("frame은 numpy array여야 합니다.")
            if not validate_plate_info(plate_info):
                raise ValueError("유효하지 않은 번호판 정보입니다.")
            
            # 번호판 정보 추출
            area = plate_info['area']
            crop_box = (area['x'], area['y'], area['width'], area['height'])
            
            # 이미지 전처리
            hsv_image = preprocess_image(frame, crop_box, area.get('angle', 0))
            
            # 특징 추출
            features = extract_features(hsv_image)
            
            # 예측
            xgb_pred = self.xgb_model.predict([features])[0]
            xgb_prob = self.xgb_model.predict_proba([features])[0][1]
            
            # 신뢰도 기반 앙상블
            if xgb_prob < self.confidence_threshold:
                prediction = self.lgb_model.predict([features])[0]
                model_used = 'lgb'
            else:
                prediction = xgb_pred
                model_used = 'xgb'
            
            elapsed_time = time.time() - start_time
            metrics = ProcessingMetrics(
                elapsed_time=elapsed_time,
                confidence_score=xgb_prob,
                model_used=model_used
            )
            
            if elapsed_time > self.max_processing_time:
                self.logger.warning(f"처리 시간 초과: {elapsed_time:.2f}초")
            
            return bool(prediction), metrics
            
        except Exception as e:
            self.logger.error(f"프레임 처리 중 오류 발생: {str(e)}")
            metrics = ProcessingMetrics(
                elapsed_time=time.time() - start_time,
                confidence_score=0.0,
                model_used='none',
                error_occurred=True,
                error_message=str(e)
            )
            raise

    def _process_frame_wrapper(self, frame: np.ndarray, plate_info: Dict) -> Tuple[bool, ProcessingMetrics]:
        """프레임 처리 래퍼 함수 (병렬 처리용)"""
        try:
            return self.process_frame(frame, plate_info)
        except Exception as e:
            self.logger.error(f"프레임 처리 실패: {str(e)}")
            return False, ProcessingMetrics(
                elapsed_time=0.0,
                confidence_score=0.0,
                model_used='none',
                error_occurred=True,
                error_message=str(e)
            )

    def process_batch(self, frames: List[np.ndarray], plate_infos: List[Dict]) -> List[Tuple[bool, ProcessingMetrics]]:
        """여러 프레임 일괄 처리
        
        Args:
            frames (List[np.ndarray]): 처리할 프레임 리스트
            plate_infos (List[Dict]): 번호판 정보 리스트
            
        Returns:
            List[Tuple[bool, ProcessingMetrics]]: (예측 결과, 처리 메트릭) 리스트
        """
        if len(frames) != len(plate_infos):
            raise ValueError("frames와 plate_infos의 길이가 일치하지 않습니다.")
        
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i+self.batch_size]
            batch_plate_infos = plate_infos[i:i+self.batch_size]
            
            # 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_frame = {
                    executor.submit(self._process_frame_wrapper, frame, plate_info): (frame, plate_info)
                    for frame, plate_info in zip(batch_frames, batch_plate_infos)
                }
                
                for future in as_completed(future_to_frame):
                    frame, plate_info = future_to_frame[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.metrics_history.append(result[1])
                    except Exception as e:
                        self.logger.error(f"배치 처리 중 오류 발생: {str(e)}")
                        results.append((False, ProcessingMetrics(
                            elapsed_time=0.0,
                            confidence_score=0.0,
                            model_used='none',
                            error_occurred=True,
                            error_message=str(e)
                        )))
        
        return results

    def get_metrics_summary(self) -> Dict:
        """처리 메트릭 요약 정보 반환"""
        if not self.metrics_history:
            return {}
            
        return {
            'total_processed': len(self.metrics_history),
            'avg_processing_time': np.mean([m.elapsed_time for m in self.metrics_history]),
            'avg_confidence': np.mean([m.confidence_score for m in self.metrics_history]),
            'error_rate': sum(1 for m in self.metrics_history if m.error_occurred) / len(self.metrics_history),
            'model_usage': {
                'xgb': sum(1 for m in self.metrics_history if m.model_used == 'xgb'),
                'lgb': sum(1 for m in self.metrics_history if m.model_used == 'lgb')
            }
        } 