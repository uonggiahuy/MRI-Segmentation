import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model

# Load mô hình ResNet-50
def load_classification_model():
    with open('model/Classify(new)/resnet-50-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    model = tf.keras.models.model_from_json(json_savedModel, custom_objects={'Model': Model})
    model.load_weights('model/Classify(new)/weights.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

# Tiền xử lý ảnh
def preprocess_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {file_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
    return np.expand_dims(img_normalized, axis=0)

# Benchmark toàn bộ thư mục test
def benchmark(model, test_dir):
    y_true = []
    y_pred = []
    error_files = []

    label_map = {
        "no_tumor": 0,
        "tumor": 1
    }

    for label_name, label_value in label_map.items():
        folder_path = os.path.join(test_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Không tìm thấy thư mục: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                image = preprocess_image(file_path)
                prediction = model.predict(image)
                pred_class = np.argmax(prediction[0])
                y_true.append(label_value)
                y_pred.append(pred_class)
            except Exception as e:
                error_files.append(file_path)
                print(f"Lỗi xử lý {file_path}: {e}")

    return y_true, y_pred, error_files

# Run benchmark
if __name__ == "__main__":
    test_dir = input("Nhập đường dẫn đến thư mục test (VD: test/): ").strip()
    if not os.path.exists(test_dir):
        print("Thư mục không tồn tại.")
        exit()

    print("Đang tải mô hình phân loại...")
    model = load_classification_model()

    print("Đang benchmark toàn bộ tập test...")
    y_true, y_pred, errors = benchmark(model, test_dir)

    print("\nĐÁNH GIÁ MÔ HÌNH:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["KHÔNG CÓ KHỐI U", "CÓ KHỐI U"]))

    if errors:
        print(f"\nCó {len(errors)} ảnh bị lỗi khi xử lý:")
        for f in errors:
            print(f" - {f}")
