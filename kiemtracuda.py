from tensorflow.python.platform import build_info as tf_build_info
print(tf_build_info.cuda_version_number)  # CUDA version TensorFlow expects
print(tf_build_info.cudnn_version_number)  # cuDNN version TensorFlow expects
