from typing import Tuple, Optional
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

class PizzaDetector:
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        dropout_rate: float = 0.5
    ):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        
    def build_model(self, fine_tune_layers: Optional[int] = None) -> Model:
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        if fine_tune_layers:
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        # First Dense layer
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Second Dense layer
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

def create_model() -> Model:
    detector = PizzaDetector()
    return detector.build_model(fine_tune_layers=50)