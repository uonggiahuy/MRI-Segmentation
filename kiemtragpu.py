import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
#cài phiên bản tensorflow 2.18 và keras 3.8 (có sẵn với tensorflow 2.18)
#cài cuda 12.5 và cudnn 9.3