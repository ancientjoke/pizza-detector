from typing import Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import numpy as np

class DataPreprocessor:
    @staticmethod
    def create_data_generators() -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **config.AUGMENTATION_CONFIG,
            preprocessing_function=DataPreprocessor._preprocess_input
        )

        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=DataPreprocessor._preprocess_input
        )

        return train_datagen, validation_datagen

    @staticmethod
    def _preprocess_input(image: np.ndarray) -> np.ndarray:
        """Custom preprocessing function for images"""
        # Convert to float32 if not already
        image = tf.cast(image, tf.float32)
        
        # Normalize to [-1, 1] range
        image = (image - 127.5) / 127.5
        
        # Apply color augmentation
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        return image

def create_data_generators() -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    return DataPreprocessor.create_data_generators()