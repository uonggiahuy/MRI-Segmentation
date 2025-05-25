import os
import cv2
import numpy as np
import tensorflow as tf

def load_classification_model():
    classify_model = tf.keras.models.load_model('/home/choconadyne/Documents/MRI_segmentation_khnt/model/classify218/classifier-resnet-weights.keras')
    return classify_model

def load_and_preprocess_image(file_path):
    print(f"Đang xử lý ảnh: {file_path}")
    img_original = cv2.imread(file_path)
    if img_original is None:
        raise ValueError("Không thể đọc ảnh. Vui lòng kiểm tra lại đường dẫn hoặc định dạng.")
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_rgb, img_batch

def predict_classification_only(file_path, classify_model):
    img_rgb, img_batch = load_and_preprocess_image(file_path)
    prediction = classify_model.predict(img_batch)
    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class] * 100
    label = "CÓ KHỐI U" if pred_class == 1 else "KHÔNG CÓ KHỐI U"

    return label, confidence

# Test trực tiếp
if __name__ == "__main__":
    file_path = input("Nhập đường dẫn ảnh MRI của bạn: ")
    if not os.path.exists(file_path):
        print("File không tồn tại.")
        exit()

    print("Đang tải mô hình phân loại...")
    classify_model = load_classification_model()

    label, confidence = predict_classification_only(file_path, classify_model)
    print("\nKẾT QUẢ PHÂN LOẠI:")
    print(f"Dự đoán: {label}")
    print(f"Độ tin cậy: {confidence:.2f}%")
