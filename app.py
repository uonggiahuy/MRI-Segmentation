from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
import uuid
from pathlib import Path
from segment_classify import load_models, predict_image

app = Flask(__name__)
uploads_dir = Path("static/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = uploads_dir

# Tải mô hình
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

    # Tạo tên file duy nhất
    unique_id = str(uuid.uuid4())[:8]
    file_ext = os.path.splitext(file.filename)[-1].lower()

    if file_ext in ['.tif', '.tiff']:
        img = Image.open(file.stream)
        filename = f"input_{unique_id}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(file_path, format='PNG')
    else:
        filename = f"input_{unique_id}{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    if models_loaded:
        try:
            label, confidence, overlay_path = predict_image(file_path, segment_model, classify_model)
            overlay_filename = os.path.basename(overlay_path) if overlay_path else None
            label = f"{label} (Độ tin cậy: {confidence:.2f}%)"
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            label = "Lỗi xử lý ảnh. Vui lòng thử lại."
            overlay_filename = None
    else:
        label = "Mô hình chưa được tải."
        overlay_filename = None

    return render_template('index.html', 
                           filename=filename, 
                           overlay_filename=overlay_filename, 
                           label=label)

if __name__ == '__main__':
    app.run(debug=True)
