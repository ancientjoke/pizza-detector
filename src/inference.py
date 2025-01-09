from typing import Union, Tuple
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PizzaPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            model_path = Path(config.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = load_model(str(model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, config.IMAGE_SIZE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image_path: Union[str, Path]) -> Tuple[float, str]:
        try:
            img = self._preprocess_image(image_path)
            prediction = self.model.predict(img, verbose=0)[0][0]
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            label = "Pizza" if prediction >= 0.5 else "Not Pizza"
            return confidence, label
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

def predict_image(image_path: Union[str, Path]) -> Tuple[float, str]:
    predictor = PizzaPredictor()
    return predictor.predict(image_path)