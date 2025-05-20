# File test trực tiếp và hiển thị với matplotlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.keras.models import Model
from utilities import focal_tversky, tversky


# Tải mô hình phân loại (ResNet50)
print("Đang tải mô hình phân loại...")
try:
    with open('/home/choconadyne/Documents/MRI_segmentation_khnt/model/Classify(new)/resnet-50-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    model = tf.keras.models.model_from_json(json_savedModel, custom_objects={'Model': Model})
    model.load_weights('/home/choconadyne/Documents/MRI_segmentation_khnt/model/Classify(new)/weights.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    print("Đã tải mô hình phân loại thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình phân loại: {e}")

# Tải mô hình phân đoạn (ResUNet)
print("Đang tải mô hình phân đoạn...")
try:
    with open('/home/choconadyne/Documents/MRI_segmentation_khnt/model/Segments(new)/ResUNet-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    model_seg = tf.keras.models.model_from_json(json_savedModel,
                                                custom_objects={'Model': Model,
                                                                'focal_tversky': focal_tversky,
                                                                'tversky': tversky})
    model_seg.load_weights('/home/choconadyne/Documents/MRI_segmentation_khnt/model/Segments(new)/weights_seg.hdf5')
    adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
    model_seg.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])
    print("Đã tải mô hình phân đoạn thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình phân đoạn: {e}")

# Hàm tiền xử lý ảnh
def load_and_preprocess_image(file_path):
    img_original = cv2.imread(file_path)
    
    if img_original is None:
        raise ValueError("Không thể đọc ảnh. Vui lòng kiểm tra lại đường dẫn hoặc định dạng ảnh.")

    # Hiển thị kích thước ảnh gốc (tùy chọn để debug)
    print(f"Kích thước ảnh gốc: {img_original.shape}")

    # Đổi sang RGB
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Resize ảnh về 256x256
    img_resized = cv2.resize(img_rgb, (256, 256))

    # Chuẩn hóa cường độ (z-score normalization)
    img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)

    # Thêm batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_rgb, img_batch

# Hàm dự đoán và trực quan hóa
def predict_and_visualize(img_rgb, img_batch):
    """
    Dự đoán và trực quan hóa kết quả

    Parameters:
    img_rgb (ndarray): Ảnh RGB gốc
    img_batch (ndarray): Ảnh đã chuẩn bị cho mô hình
    """
    # Dự đoán bằng mô hình phân loại
    classification_prediction = model.predict(img_batch)

    # Xác định lớp dự đoán (0: không có khối u, 1: có khối u)
    pred_class = np.argmax(classification_prediction[0])

    # Tính độ tin cậy của dự đoán
    confidence = classification_prediction[0][pred_class] * 100

    # Kết quả dự đoán phân loại
    class_result = "CÓ KHỐI U" if pred_class == 1 else "KHÔNG CÓ KHỐI U"

    print(f"\n Kết quả phân tích ảnh MRI:")
    print(f" Dự đoán: {class_result}")
    print(f" Độ tin cậy: {confidence:.2f}%")

    # Nếu dự đoán có khối u, thực hiện phân đoạn
    if pred_class == 1:
        print("\nĐang phân đoạn vị trí khối u...")
        # Dự đoán bằng mô hình phân đoạn
        segmentation_prediction = model_seg.predict(img_batch)

        # Chuyển đổi dự đoán từ xác suất sang nhị phân (0 hoặc 1)
        predicted_mask = segmentation_prediction[0].squeeze().round()

        # Trực quan hóa kết quả
        plt.figure(figsize=(15, 10))

        # Hiển thị ảnh MRI gốc
        plt.subplot(1, 3, 1)
        plt.title("Ảnh MRI gốc")
        plt.imshow(img_rgb)
        plt.axis('off')

        # Hiển thị kết quả phân đoạn (mặt nạ)
        plt.subplot(1, 3, 2)
        plt.title("Phân đoạn khối u")
        plt.imshow(predicted_mask, cmap='gray')
        plt.axis('off')

        # Resize predict_mask về đúng kích thước ảnh gốc
        mask_resized = cv2.resize(predicted_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Hiển thị ảnh MRI với phân đoạn khối u chồng lên
        overlay_img = img_rgb.copy()
        # Áp dụng màu xanh lá cho vùng khối u
        overlay_img[mask_resized == 1] = [0, 255, 0]

        plt.subplot(1, 3, 3)
        plt.title("Ảnh MRI với phân đoạn")
        plt.imshow(overlay_img)
        plt.axis('off')

        plt.tight_layout()
        plt.suptitle(f"Kết quả dự đoán: {class_result} (Độ tin cậy: {confidence:.2f}%)", fontsize=16)
        plt.show()

        # Tính tỷ lệ vùng khối u
        tumor_percentage = (np.sum(predicted_mask) / predicted_mask.size) * 100
        print(f" Kích thước khối u: {tumor_percentage:.2f}% diện tích ảnh")

    else:
        # Chỉ hiển thị ảnh MRI gốc với kết quả phân loại
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title(f"Kết quả dự đoán: {class_result} (Độ tin cậy: {confidence:.2f}%)")
        plt.axis('off')
        plt.show()

    return class_result, confidence


file_path = input("Nhập đường dẫn ảnh MRI của bạn (VD: mri_image.jpg): ")

if not os.path.exists(file_path):
    print("File không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Tiền xử lý ảnh
img_rgb, img_batch = load_and_preprocess_image(file_path)

# Dự đoán và hiển thị kết quả
result, confidence = predict_and_visualize(img_rgb, img_batch)
