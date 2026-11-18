"""cool-roof-detection 실험 유틸리티.

0단계(데이터 파이프라인·기준선 확립)에 필요한 기능을 모듈별로 분리했다.

- data_check : 클래스 ID 매핑 확인·리매핑, 이미지–라벨 정합성 검사
- dedup      : 지각 해시(dHash) 기반 스플릿 간 근접 중복 이미지 탐지
- splits     : Validation 이미지 목록 고정, 라벨 버전(val_v1, val_v2 …) 관리
- experiment : seed 반복 학습·평가, 결과 CSV 자동 기록, 평균±표준편차 요약
"""
