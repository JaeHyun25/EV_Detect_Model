# EV Detector Core

실시간 이미지 처리를 통한 전기차 감지 시스템

## 주요 기능

- 실시간 이미지 처리를 통한 전기차 여부 판정
- 번호판 정보와 이미지 데이터 기반 분석
- 에러 케이스 및 불확실한 판정 결과 저장
- 설정 기반의 유연한 시스템 운영

## 시스템 요구사항

- Python 3.8 이상
- OpenCV
- NumPy
- XGBoost
- LightGBM
- PyYAML

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/ev-detector-core.git
cd ev-detector-core
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 설정 파일 수정
   - `config/config.yaml` 파일에서 필요한 설정 변경

2. 실행
```bash
python main.py
```

## 입력 데이터 형식

### 이미지 데이터
- NumPy array 형식
- 크기: (1080, 1920, 3)
- 채널: BGR

### JSON 데이터
```json
{
    "area": {
        "angle": float,
        "height": int,
        "width": int,
        "x": int,
        "y": int
    },
    "attrs": {
        "ev": bool
    },
    "conf": {
        "ocr": float,
        "plate": float
    },
    "text": str
}
```

## 출력 데이터 형식

```json
{
    "area": {
        "angle": float,
        "height": int,
        "width": int,
        "x": int,
        "y": int
    },
    "attrs": {
        "ev": bool
    },
    "conf": {
        "ocr": float,
        "plate": float,
        "ev": float
    },
    "elapsed": float,
    "ev": bool,
    "text": str,
    "timestamp": str
}
```

## 디렉토리 구조

```
ev_detector_core/
├── main.py              # 메인 실행 파일
├── config/
│   └── config.yaml     # 설정 파일
├── src/
│   ├── detector/       # 전기차 감지 모듈
│   ├── models/        # 학습된 모델 파일
│   └── utils/         # 유틸리티 함수
├── logs/              # 로그 파일
├── error_cases/       # 에러 케이스
└── uncertain_cases/   # 불확실한 판정 결과
```

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 