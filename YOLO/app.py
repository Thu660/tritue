from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)

# Thư mục chứa ảnh upload và kết quả
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load mô hình YOLO đã huấn luyện
model = YOLO(os.path.join(os.path.dirname(__file__), "yolo11s.pt"))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Nhận diện đối tượng bằng YOLO
        results = model(filepath)
        result_image = results[0].plot()
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        Image.fromarray(result_image).save(result_path)

        return render_template('index.html',
                               original=filename,
                               result=result_filename)

    return render_template('index.html', original=None, result=None)

# Route trả về ảnh trong thư mục uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename)

if __name__ == "__main__":
    app.run(debug=True)
