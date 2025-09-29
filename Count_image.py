import pandas as pd
from pathlib import Path

# --------------------------------------------------
# ✅ 설정 부분
# --------------------------------------------------
# 스크립트가 있는 폴더를 기준으로 경로 설정
# (예: '.../code/try/' 폴더에 이 스크립트가 있다고 가정)
BASE_DIR = Path(__file__).resolve().parent

# 이미지들이 들어있는 상위 폴더 경로 ('.../images/')
IMAGES_DIR = BASE_DIR.parent / "images"

# 결과를 저장할 CSV 파일 경로 및 이름
CSV_SAVE_PATH = BASE_DIR / "종별이미지개수.csv"
# --------------------------------------------------


def count_images_and_save_csv():
    """
    지정된 폴더 내의 각 하위 폴더에 있는 파일 개수를 세어 CSV로 저장합니다.
    """
    # 결과를 저장할 리스트
    image_counts = []

    # images 폴더가 존재하는지 확인
    if not IMAGES_DIR.is_dir():
        print(f"❌ 오류: '{IMAGES_DIR}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f" сканирование '{IMAGES_DIR.name}' 폴더에서 이미지 개수를 스캔합니다...")

    # images 폴더 내의 모든 항목을 순회
    for class_folder in IMAGES_DIR.iterdir():
        # 폴더인 경우에만 개수 세기 진행
        if class_folder.is_dir():
            folder_name = class_folder.name
            
            # 해당 폴더 내의 파일 개수 계산 (숨김 파일 등 제외)
            # '.jpg', '.jpeg', '.png' 등 일반적인 이미지 확장자만 카운트할 수 있습니다.
            # 여기서는 간단하게 모든 파일을 카운트합니다.
            file_count = sum(1 for f in class_folder.glob('*.*') if f.is_file())
            
            # 결과 리스트에 추가
            image_counts.append({'folder': folder_name, 'file_count': file_count})
            print(f" - '{folder_name}' 폴더에서 {file_count}개의 파일을 찾았습니다.")

    # 데이터가 있을 경우에만 CSV 파일 생성
    if image_counts:
        # 리스트를 Pandas DataFrame으로 변환
        df = pd.DataFrame(image_counts)
        
        # 'file_count'를 기준으로 내림차순 정렬
        df = df.sort_values(by='file_count', ascending=False)
        
        try:
            # CSV 파일로 저장 (Excel에서 한글이 깨지지 않도록 'utf-8-sig' 인코딩 사용)
            df.to_csv(CSV_SAVE_PATH, index=False, encoding='utf-8-sig')
            print(f"\n✅ 결과가 '{CSV_SAVE_PATH.name}' 파일로 성공적으로 저장되었습니다.")
            
            # 상위 5개 결과 미리보기 출력
            print("\n--- 결과 미리보기 (상위 5개) ---")
            print(df.head().to_string(index=False))
            
        except Exception as e:
            print(f"\n❌ CSV 파일 저장 중 오류가 발생했습니다: {e}")
    else:
        print("\n⚠️ 스캔할 하위 폴더를 찾지 못했습니다.")


if __name__ == '__main__':
    count_images_and_save_csv()