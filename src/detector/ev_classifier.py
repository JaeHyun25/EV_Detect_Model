import cv2
import numpy as np
import joblib
import logging
from typing import List, Tuple, Dict, Union
import time
from ..utils.image_processing import preprocess_image, extract_features, validate_plate_info

class EVClassifier:
    def __init__(self, xgb_model_path: str, lgb_model_path: str):
        """전기차 판별 모델 로드"""
        self.xgb_model = joblib.load(xgb_model_path)
        self.lgb_model = joblib.load(lgb_model_path)
        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame: np.ndarray, plate_info: Dict) -> Tuple[bool, float]:
        """단일 프레임 처리 및 예측"""
        try:
            start_time = time.time()
            
            # 번호판 정보 유효성 검사
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
            if xgb_prob < 0.45:
                prediction = self.lgb_model.predict([features])[0]
            else:
                prediction = xgb_pred
            
            elapsed_time = time.time() - start_time
            return bool(prediction), elapsed_time
            
        except Exception as e:
            self.logger.error(f"프레임 처리 중 오류 발생: {str(e)}")
            raise

    def process_batch(self, frames: List[np.ndarray], plate_infos: List[Dict]) -> List[Tuple[bool, float]]:
        """여러 프레임 일괄 처리"""
        results = []
        for frame, plate_info in zip(frames, plate_infos):
            result = self.process_frame(frame, plate_info)
            results.append(result)
        return results 