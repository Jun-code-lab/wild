import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
from pathlib import Path
import threading

# --------------------------------------------------
# ✨ 프로그램 설정
# --------------------------------------------------
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"  # 윈도우 맑은 고딕 볼드
DEFAULT_FONT_SIZE = 14
PREDICTION_IMAGE_SIZE = (600, 600)  # GUI에 표시될 이미지 최대 크기

# --------------------------------------------------
# 🎯 핵심 예측 로직 (기존 코드 기반)
# --------------------------------------------------
def perform_prediction(model_path, image_path):
    """YOLO 모델로 이미지를 예측하고, 결과 텍스트와 시각화된 이미지를 반환합니다."""
    try:
        model = YOLO(model_path)
        true_label = image_path.parent.name
        
        results = model.predict(image_path, verbose=False)
        result = results[0]

        # 텍스트 결과 생성
        pred_label = model.names[result.probs.top1]
        pred_confidence = result.probs.top1conf.item()
        is_correct = (true_label == pred_label)
        
        result_lines = []
        result_lines.append(f"실제 라벨: '{true_label}'")
        result_lines.append("-" * 30)
        result_lines.append(f"▶ 예측 결과: '{pred_label}' ({pred_confidence*100:.2f}%)")
        result_lines.append(f"▶ 정답 여부: {'✅ 맞음' if is_correct else '❌ 틀림'}")
        result_lines.append("\n--- Top-5 예측 ---")

        top5_indices = result.probs.top5
        top5_confs = result.probs.top5conf
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
            result_lines.append(f"{i+1}. {model.names[idx]} ({conf.item()*100:.1f}%)")
        
        result_text = "\n".join(result_lines)
        
        # 이미지 시각화
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(FONT_PATH, 20)
        except IOError:
            font = ImageFont.load_default()

        box_y = 10
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
            class_name = model.names[idx]
            text = f"{i+1}. {class_name} ({conf.item()*100:.1f}%)"
            text_color = "lime" if class_name == true_label else "white"
            text_bbox = draw.textbbox((10, box_y), text, font=font)
            bg_bbox = [text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5]
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 128))
            draw.text((10, box_y), text, font=font, fill=text_color)
            box_y += text_bbox[3] - text_bbox[1] + 10
            
        return result_text, img

    except Exception as e:
        return f"오류 발생:\n{e}", None

# --------------------------------------------------
# 💻 GUI 애플리케이션 클래스
# --------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 이미지 분류기")
        self.root.geometry("1000x700")

        self.model_path = tk.StringVar()
        self.image_path = tk.StringVar()

        # --- 위젯 생성 ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(main_frame, text="파일 선택", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="모델 파일 선택 (.pt)", command=self.select_model).pack(side=tk.LEFT, padx=5)
        ttk.Label(file_frame, textvariable=self.model_path, wraplength=800).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        image_file_frame = ttk.LabelFrame(main_frame, text="이미지 선택", padding="10")
        image_file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(image_file_frame, text="이미지 파일 선택", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Label(image_file_frame, textvariable=self.image_path, wraplength=800).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 실행 버튼
        self.predict_button = ttk.Button(main_frame, text="예측 실행", command=self.start_prediction)
        self.predict_button.pack(fill=tk.X, pady=10)

        # 결과 표시 프레임
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=10, width=40, font=("맑은 고딕", 12))
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_label = ttk.Label(result_frame, text="예측 결과 이미지가 여기에 표시됩니다.", style="TLabel")
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def select_model(self):
        path = filedialog.askopenfilename(title="모델 .pt 파일을 선택하세요", filetypes=[("PyTorch Model", "*.pt")])
        if path:
            self.model_path.set(path)

    def select_image(self):
        path = filedialog.askopenfilename(title="이미지 파일을 선택하세요", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path.set(path)

    def start_prediction(self):
        if not self.model_path.get() or not self.image_path.get():
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "오류: 모델과 이미지 파일을 모두 선택해야 합니다.")
            return

        self.predict_button.config(state=tk.DISABLED, text="예측 중...")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "모델 로딩 및 예측 중...")
        
        # GUI가 멈추지 않도록 별도의 스레드에서 예측 실행
        threading.Thread(target=self.run_prediction_thread, daemon=True).start()
        
    def run_prediction_thread(self):
        model_p = Path(self.model_path.get())
        image_p = Path(self.image_path.get())
        
        result_text, visualized_img = perform_prediction(model_p, image_p)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        
        if visualized_img:
            visualized_img.thumbnail(PREDICTION_IMAGE_SIZE)
            photo = ImageTk.PhotoImage(visualized_img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 참조 유지
        
        self.predict_button.config(state=tk.NORMAL, text="예측 실행")

# --------------------------------------------------
# 🚀 프로그램 실행
# --------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()