from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Đường dẫn đến mô hình YOLOv8 đã huấn luyện (thay đổi nếu cần)
MODEL_PATH = 'yolo11s.pt'
model = None
try:
    model = YOLO(MODEL_PATH)
    print(f"Mô hình YOLO đã được tải thành công từ: {MODEL_PATH}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy mô hình tại {MODEL_PATH}")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")

def detect_objects(image_bytes):
    """Hàm thực hiện nhận diện đối tượng bằng YOLO."""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = model(img_cv2)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.int()
            confidences = r.boxes.conf
            class_ids = r.boxes.cls.int()
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                label = model.names[class_id]
                confidence_str = f'{confidence:.2f}'
                detections.append({'box': [x1, y1, x2, y2], 'label': label, 'confidence': confidence_str})
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv2, f'{label} {confidence_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, img_encoded = cv2.imencode('.png', img_cv2)
        base64_image = base64.b64encode(img_encoded).decode('utf-8')
        return base64_image, detections
    except Exception as e:
        print(f"Lỗi trong hàm detect_objects: {e}")
        return None, []

@app.route('/', methods=['GET'])
def index():
    """Hiển thị trang chủ để tải ảnh lên."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý việc tải ảnh lên và thực hiện nhận diện."""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được tải lên'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'})
    if file and model:
        try:
            image_bytes = file.read()
            base64_image, detections = detect_objects(image_bytes)
            return render_template('result_colab.html', image=f'data:image/png;base64,{base64_image}', detections=detections)
        except Exception as e:
            return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'})
    elif not model:
        return jsonify({'error': 'Mô hình YOLO chưa được tải'})
    return render_template('result.html', image=None, detections=[])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))