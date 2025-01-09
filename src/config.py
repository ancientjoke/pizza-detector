from typing import Tuple, Dict
from pathlib import Path

IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
EPOCHS: int = 20
LEARNING_RATE: float = 0.001

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH: str = str(BASE_DIR / "models/saved_models/pizza_detector.h5")

TRAIN_CONFIG: Dict = {
    'validation_split': 0.2,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'min_lr': 1e-6
}

AUGMENTATION_CONFIG: Dict = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'zoom_range': 0.15
}