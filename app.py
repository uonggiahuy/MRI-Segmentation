from flask import Flask, render_template, request, redirect, url_for
import os
from pathlib import Path
from utils import load_models, predict_image

app = Flask(__name__)
# Đảm bảo thư mục uploads tồn tại
uploads_dir = Path("static/uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

app.config['UPLOAD_FOLDER'] = uploads_dir

# Load model ngay khi khởi động
try:
    segment_model, classify_model = load_models()
    models_loaded = True
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    models_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Lưu file ảnh
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if models_loaded:
        try:
            # Dự đoán bằng mô hình
            label, overlay_filename = predict_image(file_path, segment_model, classify_model)
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            label = "Lỗi xử lý ảnh. Vui lòng thử lại."
            overlay_filename = None
    else:
        label = "Mô hình chưa được tải. Vui lòng kiểm tra lại cấu hình."
        overlay_filename = None

    return render_template('index.html', 
                          filename=filename, 
                          overlay_filename=overlay_filename, 
                          label=label)

if __name__ == '__main__':
    app.run(debug=True)