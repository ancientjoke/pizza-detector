import cv2
import os

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_directories():
    dirs = [
        'data/training/pizza',
        'data/training/not_pizza',
        'data/validation/pizza',
        'data/validation/not_pizza',
        'data/test/pizza',
        'data/test/not_pizza',
        'models/saved_models'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)