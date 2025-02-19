import tensorflow as tf
import gc

def clear_gpu_memory():
    """Clear GPU memory and perform garbage collection"""
    tf.keras.backend.clear_session()
    gc.collect()
    # Optional: Force TensorFlow to release GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass