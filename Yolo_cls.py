from ultralytics import YOLO
from pathlib import Path
import torch
import os
# --------------------------------------------------
# ✅ 경로 설정 (수정 없음)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"
SAVE_PATH = BASE_DIR.parent / "runs" # '.../CODE/runs'를 가리킴

# --------------------------------------------------
# ✅ 옵션 설정
# --------------------------------------------------
EXPERIMENT_BASE_NAME = 'test11'
EARLY_STOPPING_PATIENCE = 10
# ✨ [추가] 실시간 데이터 증강 옵션
# 추천 1: 균형 잡힌 강화
AUGMENTATION_OPTIONS = {
    'degrees': 25,         # 회전 각도 범위 증가
    'translate': 0.15,       # 이동 비율 증가
    'scale': 0.15,         # 크기 조절 비율 증가
    'shear': 5,            # 이미지 찌그러뜨리기 추가
    'fliplr': 0.5,         # 좌우 반전은 유지
    'mosaic': 1.0,         # Mosaic는 유지
    'mixup': 0.3,          # Mixup 확률 증가
    
    # --- 색상 증강 추가 (매우 중요) ---
    'hsv_h': 0.015,        # 색상(Hue) 변화 범위
    'hsv_s': 0.7,          # 채도(Saturation) 변화 범위
    'hsv_v': 0.4           # 명도(Value) 변화 범위
}
# --------------------------------------------------



def get_next_experiment_name(project_dir: Path, base_name: str) -> str:
    """
    지정된 프로젝트 폴더를 확인하여 다음 실험 이름을 반환합니다.
    먼저 base_name 자체를 확인하고, 존재하면 숫자를 붙여나갑니다.
    """
    # 1. 기본 이름(base_name) 자체를 먼저 확인
    base_exp_path = project_dir / "classify" / base_name
    if not base_exp_path.exists():
        return base_name  # 기본 이름이 비어있으면 그대로 사용

    # 2. 기본 이름이 이미 존재하면, 뒤에 숫자를 붙여서 다음 번호 찾기
    i = 1
    while True:
        exp_name = f"{base_name}{i}"
        if not (project_dir / "classify" / exp_name).exists():
            return exp_name
        i += 1


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("경고: CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다.")

    print("--- 1단계: 모델 학습을 시작합니다 ---")

    model = YOLO('yolov8n-cls.pt')

    # ✨ [수정됨] 함수에 SAVE_PATH를 직접 전달
    next_experiment_name = get_next_experiment_name(SAVE_PATH, EXPERIMENT_BASE_NAME)
    print(f"이번 학습 결과는 '{SAVE_PATH}/classify/{next_experiment_name}' 폴더에 저장됩니다.")

    results = model.train(
        data=DATASET_PATH,
        epochs=75,
        imgsz=384,
        # ✨✨✨ [가장 중요] project 옵션 추가! ✨✨✨
        project=SAVE_PATH,
        # name에는 하위 폴더 이름만 지정
        name=f"classify/{next_experiment_name}",
        verbose=False,
        batch=32,
        patience=EARLY_STOPPING_PATIENCE,
        lr0=0.01,
        **AUGMENTATION_OPTIONS,
        weight_decay=0.001, # 가중치 조절하기 
        device=0,
        workers=min(16, os.cpu_count()), # cpu 병목 현상 처리(쓰레드에 일 전부 줘서 빨리 해결하기)
        cashe = True # cpu 병목 현상 처리
        
    )

    print("\n--- 학습 완료! ---")