# convert_model.py
from keras.models import load_model as keras_load_model
import tensorflow as tf

model = keras_load_model(r"E:\Code programs\n2025\MRI segmentation khnt\model\classifier-resnet-weights.keras")
tf.keras.models.save_model(model, r"E:\Code programs\n2025\MRI segmentation khnt\model\tf-classifier-resnet-weights.keras")

print("Đã chuyển đổi mô hình.")
