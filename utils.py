import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path

def load_models():
    segment_model = None
    classify_model = None

    try:
        segment_model = tf.keras.models.load_model(
            Path("E:/Code programs/n2025/MRI segmentation khnt/model/weights_seg-1.hdf5"),
            compile=False
        )
        print("Done segment model")
    except Exception as e:
        print("Error segment model") 
        print(e)

    try:
        classify_model = tf.keras.models.load_model(
            Path("E:/Code programs/n2025/MRI segmentation khnt/model/classifier-resnet-weights.keras"),
            compile=False
        )
        print("Done classify model")
    except Exception as e:
        print("Error classify model")
        print(e)

    return segment_model, classify_model


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_array = np.array(img) / 255.0
    return img_array[np.newaxis, ...], img

def save_overlay(original_img, mask, filename):
    plt.figure(figsize=(4,4))
    plt.imshow(original_img)
    plt.imshow(mask, cmap='Reds', alpha=0.4)
    plt.axis('off')
    
    overlay_path = os.path.join('static/uploads', 'overlay_' + filename)
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return 'overlay_' + filename

def predict_image(img_path, segment_model, classify_model):
    img_array, original_img = preprocess_image(img_path)

    # Phân đoạn
    pred_mask = segment_model.predict(img_array)[0]
    mask = (pred_mask > 0.5).astype(np.uint8)

    # Phân loại
    class_pred = classify_model.predict(img_array)
    class_label = np.argmax(class_pred)
    label_map = ['Không có khối u', 'U lành tính', 'U ác tính']
    label = label_map[class_label]

    overlay_filename = save_overlay(original_img, mask, os.path.basename(img_path))

    return label, overlay_filename

if __name__ == "__main__":
    # Đường dẫn ảnh MRI đầu vào cần test
    test_img_path = r"E:\Code programs\n2025\MRI segmentation khnt\static\uploads\clipboard-image.png"  # <-- đổi thành ảnh bạn có thật

    # Kiểm tra ảnh có tồn tại không
    if not os.path.exists(test_img_path):
        print(f"Ảnh test không tồn tại: {test_img_path}")
    else:
        # Load mô hình
        segment_model, classify_model = load_models()

        # Dự đoán
        label, overlay = predict_image(test_img_path, segment_model, classify_model)

        # In kết quả
        print(f"\n🧠 Kết quả phân loại: {label}")
        print(f"📁 Ảnh overlay đã lưu: static/uploads/{overlay}")

