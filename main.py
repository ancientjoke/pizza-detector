import argparse
import logging
from pathlib import Path
from src.train import train_model
from src.inference import predict_image
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Pizza Detector CLI')
    parser.add_argument(
        '--mode',
        choices=['train', 'predict'],
        required=True,
        help='Choose whether to train the model or make predictions'
    )
    parser.add_argument(
        '--image_path',
        type=Path,
        help='Path to image for prediction'
    )
    return parser

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            logger.info("Starting model training...")
            history = train_model()
            logger.info("Training completed successfully!")
            
        elif args.mode == 'predict':
            if args.image_path is None:
                logger.error("Please provide an image path for prediction")
                sys.exit(1)
                
            if not args.image_path.exists():
                logger.error(f"Image file not found: {args.image_path}")
                sys.exit(1)
                
            confidence, label = predict_image(args.image_path)
            logger.info(f"Prediction: {label}")
            logger.info(f"Confidence: {confidence:.2f}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()