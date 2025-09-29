import os
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont 
import pandas as pd

# --------------------------------------------------
# ✅ 사용자가 수정해야 할 부분
# --------------------------------------------------
# 1. 학습된 모델(.pt 파일)의 전체 경로
MODEL_PATH = r"C:\Users\sega0\Desktop\code\runs\classify\test10\weights\best.pt"

# 2. ✨ [가장 중요] 예측하고 싶은 '이미지 파일 하나'의 전체 경로
TEST_IMAGE_PATH = r"C:\Users\sega0\Desktop\code\IMG_3892.JPG"

# 3. (선택) 결과 이미지와 CSV를 저장할 기본 폴더 경로
RESULTS_SAVE_PATH = Path(MODEL_PATH).parent.parent

# 4. (선택) 텍스트 표시에 사용할 폰트 경로
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
# --------------------------------------------------


def predict_single_image():
    """지정된 단일 이미지 파일을 예측하고 결과를 시각화하여 저장합니다."""
    
    model_path = Path(MODEL_PATH)
    image_path = Path(TEST_IMAGE_PATH)

    # 시각화된 이미지를 저장할 폴더 경로 설정 및 생성
    visualized_save_dir = RESULTS_SAVE_PATH / "single_prediction_results"
    visualized_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 또는 이미지 파일이 존재하는지 확인
    if not model_path.is_file():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요:\n -> {model_path}")
        return
    if not image_path.is_file():
        print(f"❌ 오류: 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요:\n -> {image_path}")
        return
        
    print(f"모델을 로드합니다: {model_path.name}")
    model = YOLO(model_path)
    print("✅ 모델 로드 완료.")

    # 실제 라벨을 이미지의 부모 폴더 이름으로 간주
    true_label = image_path.parent.name
    print(f"이미지 파일의 실제 라벨 (폴더명 기준): '{true_label}'")
    
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except IOError:
        print(f"⚠️ 경고: 폰트를 로드할 수 없어 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    print(f"\n{'='*50}\n▶ '{image_path.name}' 파일 예측 시작...\n{'='*50}")

    try:
        # 단일 이미지 예측
        results = model.predict(image_path, verbose=False)
        result = results[0]
        
        # Top-1 예측 결과
        pred_label = model.names[result.probs.top1]
        pred_confidence = result.probs.top1conf.item()
        is_correct = (true_label == pred_label)
        
        print(f" -> 예측 클래스: '{pred_label}'")
        print(f" -> 신뢰도: {pred_confidence*100:.2f}%")
        print(f" -> 결과: {'✅ 정답' if is_correct else '❌ 오답'}")

        # Top-5 예측 결과 시각화
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        top5_indices = result.probs.top5
        top5_confs = result.probs.top5conf
        
        box_y = 10
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
            class_name = model.names[idx]
            text = f"{i+1}. {class_name} ({conf.item()*100:.1f}%)"
            
            # 실제 라벨과 일치하는 예측은 빨간색으로 표시
            text_color = "red" if class_name == true_label else "white"
            
            # 텍스트 배경 박스 그리기
            text_bbox = draw.textbbox((10, box_y), text, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0, 128))
            draw.text((10, box_y), text, font=font, fill=text_color)
            
            box_y += text_bbox[3] - text_bbox[1] + 5

        # 시각화된 이미지 저장
        save_path = visualized_save_dir / image_path.name
        img.save(save_path)
        print(f"\n✨ 시각화된 이미지가 저장되었습니다:\n -> {save_path}")

    except Exception as e:
        print(f"❌ 예측 또는 시각화 중 오류가 발생했습니다: {e}")


if __name__ == '__main__':
    predict_single_image()