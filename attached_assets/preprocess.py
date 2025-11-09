from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(path):
    img = image.load_img(path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = img / 255.0
    return img
