import shutil
from pathlib import Path
import pandas as pd
import random
import json

# --------------------------------------------------
# ✅ 설정 부분
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# 원본 이미지 폴더
ORIGINAL_DATASET_PATH = BASE_DIR.parent / "images"
# 기본 베이스가 될, 미리 분할된 train/val 폴더
FIXED_DATA_PATH = BASE_DIR.parent / "other"
# 최종 결과물이 저장될 폴더
NEW_DATASET_PATH = BASE_DIR / "dataset"

# 추가 처리 대상을 정할 CSV 파일
CSV_FILE_PATH = BASE_DIR / "종별이미지개수.csv"
# Test 데이터를 고정할 JSON 파일
PREDEFINED_TEST_JSON_PATH = BASE_DIR / "exclude_files.json"

# 각종 옵션
TRAIN_RATIO = 0.8
NUM_TOP_CLASSES = 10
RANDOM_SEED = 42

# --------------------------------------------------

def create_combined_dataset(original_path, fixed_path, new_path, csv_path, json_path, num_top, train_ratio, random_seed):
    """
    새로운 순서에 따라 데이터셋을 생성하는 함수
    1. 'other' 폴더를 'dataset'으로 먼저 복사
    2. CSV 목록의 클래스에 대해 Test 데이터 생성 및 Train/Val 데이터 추가
    """
    random.seed(random_seed)
    
    # --- 1단계: 'other' 폴더를 'dataset'의 기본 베이스로 복사 ---
    print("--- 1단계: 'other' 폴더를 'dataset'으로 초기 복사 시작 ---")
    if new_path.exists():
        shutil.rmtree(new_path)
        print(f" - 기존 '{new_path.name}' 폴더 삭제 완료.")
        
    try:
        shutil.copytree(fixed_path, new_path)
        print(f"✅ '{fixed_path.name}' 폴더를 '{new_path.name}'(으)로 복사 완료.")
    except FileNotFoundError:
        print(f" - ⚠️ 경고: '{fixed_path.name}' 폴더를 찾을 수 없어 빈 'dataset' 폴더를 생성합니다.")
        new_path.mkdir()

    # --- 2단계 & 3단계: CSV/JSON 정보로 Test 생성 및 Train/Val 추가 ---
    print("\n--- 2 & 3단계: CSV 목록 기준으로 데이터 추가 작업 시작 ---")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            test_files_set = set(json.load(f)['exclude'])
        print(f"✅ '{json_path.name}' 파일에서 {len(test_files_set)}개의 고정 Test 이미지 목록을 불러왔습니다.")
    except Exception as e:
        print(f"❌ Test 목록 파일('{json_path.name}') 처리 중 오류: {e}")
        return

    try:
        df = pd.read_csv(csv_path)
        top_folders = df.nlargest(num_top, 'file_count')['folder'].tolist()
        print(f"✅ CSV 파일에서 추가 처리할 상위 {num_top}개 폴더를 선정했습니다.")
        print(" - 대상 폴더:", top_folders)
    except Exception as e:
        print(f"❌ CSV 파일 처리 중 오류: {e}")
        return

    train_path = new_path / 'train'
    val_path = new_path / 'val'
    test_path = new_path / 'test'
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    for class_name in top_folders:
        print(f"\n▶ '{class_name}' 클래스 추가 처리 중...")
        class_dir = original_path / class_name
        
        if not class_dir.is_dir():
            print(f" - ⚠️ 경고: 원본 폴더 '{class_dir}'를 찾을 수 없어 건너뜁니다.")
            continue

        # train/val/test 각 클래스 폴더가 없으면 생성 (덮어쓰기가 아닌 추가이므로)
        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
        (test_path / class_name).mkdir(exist_ok=True)

        # Test 데이터 분리
        all_images = list(class_dir.glob('*.*'))
        predefined_test_images = [img for img in all_images if img.name in test_files_set]
        remaining_images = [img for img in all_images if img.name not in test_files_set]
        
        # Test 이미지 복사
        for img in predefined_test_images:
            shutil.copy(img, test_path / class_name)

        # 나머지 이미지로 Train/Val 데이터 추가 분할
        random.shuffle(remaining_images)
        split_point = int(len(remaining_images) * train_ratio)
        train_images = remaining_images[:split_point]
        val_images = remaining_images[split_point:]

        for img in train_images:
            shutil.copy(img, train_path / class_name)
        for img in val_images:
            shutil.copy(img, val_path / class_name)
        
        print(f" - ✅ 처리 완료: Test({len(predefined_test_images)}개), Train({len(train_images)}개), Val({len(val_images)}개) 이미지 추가.")

# --------------------------------------------------

if __name__ == '__main__':
    create_combined_dataset(
        ORIGINAL_DATASET_PATH,
        FIXED_DATA_PATH,
        NEW_DATASET_PATH,
        CSV_FILE_PATH,
        PREDEFINED_TEST_JSON_PATH,
        NUM_TOP_CLASSES,
        TRAIN_RATIO,
        RANDOM_SEED
    )
    print("\n✅ 모든 작업이 완료되었습니다.")