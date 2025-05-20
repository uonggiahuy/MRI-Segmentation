import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utilities import focal_tversky, tversky

def load_models():
    # Tải mô hình phân loại
    with open('model/Classify(new)/resnet-50-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    classify_model = tf.keras.models.model_from_json(json_savedModel, custom_objects={'Model': Model})
    classify_model.load_weights('model/Classify(new)/weights.hdf5')
    classify_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    # Tải mô hình phân đoạn
    with open('model/Segments(new)/ResUNet-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    segment_model = tf.keras.models.model_from_json(json_savedModel,
                                                    custom_objects={'Model': Model,
                                                                    'focal_tversky': focal_tversky,
                                                                    'tversky': tversky})
    segment_model.load_weights('model/Segments(new)/weights_seg.hdf5')
    adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
    segment_model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

    return segment_model, classify_model

# Hàm tiền xử lý ảnh
def load_and_preprocess_image(file_path):
    img_original = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    # Chuẩn hóa cường độ
    img_normalized = (img_rgb - np.mean(img_rgb)) / np.std(img_rgb)
    img_resized = cv2.resize(img_normalized, (256, 256))
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_rgb, img_batch

# Hàm dự đoán và trực quan hóa
def predict_image(file_path, segment_model, classify_model):
    img_rgb, img_batch = load_and_preprocess_image(file_path)

    classification_prediction = classify_model.predict(img_batch)
    pred_class = np.argmax(classification_prediction[0])
    confidence = classification_prediction[0][pred_class] * 100
    class_result = "CÓ KHỐI U" if pred_class == 1 else "KHÔNG CÓ KHỐI U"

    overlay_path = None

    if pred_class == 1:
        segmentation_prediction = segment_model.predict(img_batch)
        predicted_mask = segmentation_prediction[0].squeeze().round()

        mask_resized = cv2.resize(predicted_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay_img = img_rgb.copy()
        overlay_img[mask_resized == 1] = [0, 255, 0]

        # Tạo overlay filename khác biệt
        overlay_name = f"overlay_{uuid.uuid4().hex[:8]}.png"
        overlay_path = os.path.join("static/uploads", overlay_name)

        # Ghi ảnh
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

        # Tính tỷ lệ vùng khối u
        tumor_percentage = (np.sum(predicted_mask) / predicted_mask.size) * 100
        print(f" Kích thước khối u: {tumor_percentage:.2f}% diện tích ảnh")

    return class_result, confidence, overlay_path

'''
# Test chạy thử nghiệm
if __name__ == "__main__":
    file_path = input("Nhập đường dẫn ảnh MRI của bạn (VD: mri_image.jpg): ")

    if not os.path.exists(file_path):
        print("File không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        exit()

    # Bước 1: Tải mô hình
    segment_model, classify_model = load_models()

    # Bước 2: Dự đoán ảnh
    class_result, confidence, overlay_path = predict_image(file_path, segment_model, classify_model)

    # Bước 3: In kết quả
    print(f"Kết quả chẩn đoán: {class_result}")
    print(f"Độ tin cậy: {confidence:.2f}%")

    if overlay_path:
        print(f"Ảnh đã lưu với overlay tại: {overlay_path}")
'''