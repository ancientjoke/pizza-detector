# Usage Instructions
----------------------------------------------------------

## Create a virtual environment and install requirements:
```bashCopypython -m venv venv```
```# On mac: source venv/bin/activate```
```# On Windows: venv\Scripts\activate```
```pip install -r requirements.txt```

## Prepare your dataset:
-Collect pizza and non-pizza images
-Split them into training, validation, and test sets
-Place them in the appropriate directories under ```data/```

## Train the model:
```pythonCopyfrom src.train import train_model```
```train_model()```

## Make predictions:
```pythonCopyfrom src.inference import predict_image```
```result = predict_image('path_to_image.jpg')```
```print(f"Probability of pizza: {result:.2f}")```

## For training:
bashCopypython main.py --mode train

## For prediction:
bashCopypython main.py --mode predict --image_path path/to/your/image.jpg