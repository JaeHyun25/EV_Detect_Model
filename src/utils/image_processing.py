import cv2
import numpy as np
from typing import Tuple, Dict

def preprocess_image(image: np.ndarray, crop_box: Tuple[int, int, int, int], 
                    angle: float, target_size: Tuple[int, int] = (320, 180)) -> np.ndarray:
    """이미지 전처리 (크롭, 리사이즈, 회전, HSV 변환)"""
    x, y, w, h = crop_box
    cropped = image[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size)
    
    if angle != 0:
        center = (resized.shape[1]//2, resized.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        resized = cv2.warpAffine(resized, matrix, (resized.shape[1], resized.shape[0]))
    
    return cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

def extract_features(hsv_image: np.ndarray) -> np.ndarray:
    """HSV 히스토그램 특징 추출"""
    h, s, v = cv2.split(hsv_image)
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    
    return np.r_[hist_h, hist_s, hist_v].squeeze()

def validate_plate_info(plate_info: Dict) -> bool:
    """번호판 정보 유효성 검사"""
    if not isinstance(plate_info, dict):
        return False
    
    area = plate_info.get('area', {})
    required_fields = ['x', 'y', 'width', 'height']
    
    return all(field in area for field in required_fields) 