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
    print("Mô hình đã được tải thành công!")
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

    # Lưu file gốc (để model xử lý)
    original_filename = f"original_{unique_id}{file_ext}"
    original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(original_file_path)

    # Tạo file PNG để hiển thị trên web (từ file gốc)
    display_filename = f"display_{unique_id}.png"
    display_file_path = os.path.join(app.config['UPLOAD_FOLDER'], display_filename)
    
    if file_ext in ['.tif', '.tiff', '.jpg', '.jpeg']:
        # Chuyển đổi để hiển thị
        img = Image.open(original_file_path).convert("RGB")
        img.save(display_file_path, format='PNG')
    else:
        # File đã là PNG, copy để đồng nhất tên
        img = Image.open(original_file_path).convert("RGB")
        img.save(display_file_path, format='PNG')

    if models_loaded:
        try:
            # Sử dụng FILE GỐC để model xử lý (chất lượng cao hơn)
            class_result, confidence, overlay_path, tumor_percentage = predict_image(
                original_file_path,  # ← Đây là điểm quan trọng: dùng file gốc
                segment_model, 
                classify_model
            )
            
            overlay_filename = os.path.basename(overlay_path) if overlay_path else None
            
            # Tạo label với thông tin chi tiết hơn
            if tumor_percentage > 0:
                label = f"{class_result} (Độ tin cậy: {confidence:.2f}% - Kích thước khối u: {tumor_percentage:.2f}% diện tích ảnh)"
            else:
                label = f"{class_result} (Độ tin cậy: {confidence:.2f}%)"
                
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            import traceback
            traceback.print_exc()
            label = "Lỗi xử lý ảnh. Vui lòng thử lại."
            overlay_filename = None
    else:
        label = "Mô hình chưa được tải."
        overlay_filename = None

    # Xóa file gốc sau khi xử lý xong (tiết kiệm dung lượng)
    try:
        os.remove(original_file_path)
        print(f"Đã xóa file gốc: {original_file_path}")
    except:
        pass

    return render_template('index.html', 
                           filename=display_filename,  # Hiển thị file PNG
                           overlay_filename=overlay_filename, 
                           label=label)

if __name__ == '__main__':
    app.run(debug=True)