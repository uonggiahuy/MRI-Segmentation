import os
import cv2
import uuid
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from utilities import focal_tversky, tversky

def load_models():
    # Tải mô hình phân loại
    classify_model = tf.keras.models.load_model('/home/choconadyne/Documents/MRI_segmentation_khnt/model/classify218/classifier-resnet-weights.keras', compile=False)
    classify_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    # Tải mô hình phân đoạn
    with open('/home/choconadyne/Documents/MRI_segmentation_khnt/model/Segments(new)/ResUNet-MRI.json', 'r') as json_file:
        json_savedModel = json_file.read()
    segment_model = tf.keras.models.model_from_json(json_savedModel,
                                                    custom_objects={'Model': Model,
                                                                    'focal_tversky': focal_tversky,
                                                                    'tversky': tversky})
    segment_model.load_weights('/home/choconadyne/Documents/MRI_segmentation_khnt/model/Segments(new)/weights_seg.hdf5')
    adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
    segment_model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

    return segment_model, classify_model

# Hàm tiền xử lý ảnh
def load_and_preprocess_image(file_path):
    print(f"Đang xử lý ảnh: {file_path}")
    
    # Đọc ảnh - OpenCV hỗ trợ TIF, JPG, PNG
    img_original = cv2.imread(file_path)
    
    if img_original is None:
        raise ValueError(f"Không thể đọc ảnh từ: {file_path}")
    
    print(f"Ảnh gốc - Shape: {img_original.shape}, dtype: {img_original.dtype}")
    
    # Chuyển BGR sang RGB
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa cường độ pixel
    img_normalized = (img_rgb - np.mean(img_rgb)) / (np.std(img_rgb) + 1e-8)  # Thêm epsilon tránh chia 0
    
    # Resize về kích thước model yêu cầu (256x256)
    img_resized = cv2.resize(img_normalized, (256, 256))
    
    # Thêm batch dimension
    img_batch = np.expand_dims(img_resized, axis=0)
    
    print(f"Ảnh đã xử lý - Shape: {img_batch.shape}, dtype: {img_batch.dtype}")
    
    return img_rgb, img_batch

# Hàm dự đoán và trực quan hóa
def predict_image(file_path, segment_model, classify_model):
    try:
        # Tiền xử lý ảnh
        img_rgb, img_batch = load_and_preprocess_image(file_path)

        # BƯỚC 1: Phân loại (có khối u hay không?)
        print("Đang thực hiện phân loại...")
        classification_prediction = classify_model.predict(img_batch)
        pred_class = np.argmax(classification_prediction[0])
        confidence = classification_prediction[0][pred_class] * 100
        class_result = "CÓ KHỐI U" if pred_class == 1 else "KHÔNG CÓ KHỐI U"
        
        print(f"Kết quả phân loại: {class_result} (Độ tin cậy: {confidence:.2f}%)")

        # Khởi tạo các biến mặc định
        overlay_path = None
        tumor_percentage = 0

        # BƯỚC 2: Nếu có khối u → Thực hiện phân đoạn
        if pred_class == 1:
            print("Đang thực hiện phân đoạn khối u...")
            
            segmentation_prediction = segment_model.predict(img_batch)
            predicted_mask = segmentation_prediction[0].squeeze().round()
            
            print(f"Mask shape: {predicted_mask.shape}, unique values: {np.unique(predicted_mask)}")

            # Resize mask về kích thước ảnh gốc
            mask_resized = cv2.resize(predicted_mask, (img_rgb.shape[1], img_rgb.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Tạo ảnh overlay (tô màu xanh lá cho vùng khối u)
            overlay_img = img_rgb.copy()
            overlay_img[mask_resized == 1] = [0, 255, 0]  # Màu xanh lá RGB

            # Tạo thư mục nếu chưa tồn tại
            os.makedirs("static/uploads", exist_ok=True)
            
            # Tạo tên file overlay duy nhất
            overlay_name = f"overlay_{uuid.uuid4().hex[:8]}.png"
            overlay_path = os.path.join("static/uploads", overlay_name)

            # Lưu ảnh overlay (chuyển RGB về BGR cho OpenCV)
            success = cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            
            if not success:
                print("Lỗi khi lưu ảnh overlay")
                overlay_path = None
            else:
                print(f"Đã lưu ảnh overlay tại: {overlay_path}")

            # Tính tỷ lệ vùng khối u
            tumor_percentage = (np.sum(predicted_mask) / predicted_mask.size) * 100
            print(f"Kích thước khối u: {tumor_percentage:.2f}% diện tích ảnh")
        
        else:
            print("Không phát hiện khối u, bỏ qua bước phân đoạn")

        # Trả về đủ 4 giá trị như app.py mong đợi
        return class_result, confidence, overlay_path, tumor_percentage

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        traceback.print_exc()
        # Trả về giá trị mặc định khi có lỗi
        return "LỖI XỬ LÝ", 0, None, 0

'''
# Test chạy thử nghiệm (chỉ chạy khi run trực tiếp file này)
if __name__ == "__main__":
    file_path = input("Nhập đường dẫn ảnh MRI của bạn (VD: mri_image.jpg): ")

    if not os.path.exists(file_path):
        print("File không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        exit()

    try:
        # Tải mô hình
        print("Đang tải mô hình...")
        segment_model, classify_model = load_models()
        print("Mô hình đã được tải thành công!")

        # Dự đoán ảnh
        class_result, confidence, overlay_path, tumor_percentage = predict_image(file_path, segment_model, classify_model)

        # In kết quả
        print(f"\n=== KẾT QUẢ CHẨN ĐOÁN ===")
        print(f"Kết quả: {class_result}")
        print(f"Độ tin cậy: {confidence:.2f}%")
        
        if overlay_path:
            print(f"Ảnh phân đoạn đã lưu tại: {overlay_path}")
            print(f"Kích thước khối u: {tumor_percentage:.2f}% diện tích ảnh")
        else:
            print("Không có ảnh phân đoạn (không phát hiện khối u)")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        traceback.print_exc()
        '''