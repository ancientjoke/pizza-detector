from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from src.model import create_model
from src.data_preprocessing import create_data_generators
import config
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = create_model()
        self.train_datagen, self.validation_datagen = create_data_generators()

    def _create_callbacks(self) -> list:
        callbacks = [
            ModelCheckpoint(
                config.MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=config.TRAIN_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.TRAIN_CONFIG['reduce_lr_patience'],
                min_lr=config.TRAIN_CONFIG['min_lr'],
                verbose=1
            ),
            TensorBoard(log_dir='logs')
        ]
        return callbacks

    def train(self) -> Dict[str, Any]:
        train_generator = self.train_datagen.flow_from_directory(
            'data/training',
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='binary'
        )
        
        validation_generator = self.validation_datagen.flow_from_directory(
            'data/validation',
            target_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='binary'
        )

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Train model
        logger.info("Starting model training...")
        history = self.model.fit(
            train_generator,
            epochs=config.EPOCHS,
            validation_data=validation_generator,
            callbacks=self._create_callbacks(),
            workers=4,
            use_multiprocessing=True
        )
        
        return history.history

def train_model() -> Dict[str, Any]:
    trainer = ModelTrainer()
    return trainer.train()