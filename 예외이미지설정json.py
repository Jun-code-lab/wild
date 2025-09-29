import os
import json
from pathlib import Path

# --------------------------------------------------
# ✅ 설정할 부분
# --------------------------------------------------
# 1. ✨✨✨ 파일명을 가져올 test 폴더의 '절대 경로'를 직접 지정하세요. ✨✨✨
TEST_DATASET_PATH = Path(r'C:\Users\sega0\Desktop\code\try\dataset\test')

# 2. 생성할 JSON 파일의 경로를 지정하세요.
#    (스크립트가 있는 폴더에 생성됩니다)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON_PATH = BASE_DIR / "exclude_files.json"
# --------------------------------------------------

def create_exclude_json_from_folder(target_folder, output_path):
    """지정된 폴더와 그 하위 폴더의 모든 파일명을 읽어 JSON 파일을 생성합니다."""
    
    exclude_list = []
    
    if not target_folder.is_dir():
        print(f"❌ 오류: '{target_folder}' 폴더를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    for root, _, files in os.walk(target_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                exclude_list.append(filename)

    output_data = {"exclude": sorted(list(set(exclude_list)))}

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 성공! 총 {len(output_data['exclude'])}개의 파일명을 '{output_path.name}' 파일에 저장했습니다.")

    except Exception as e:
        print(f"❌ 오류: JSON 파일 저장 중 문제가 발생했습니다 - {e}")

if __name__ == "__main__":
    create_exclude_json_from_folder(TEST_DATASET_PATH, OUTPUT_JSON_PATH)