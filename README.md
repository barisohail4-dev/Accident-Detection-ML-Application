# Road Accident Detection Using Deep Learning

## Project Overview
This project builds a binary image classifier that detects whether a road scene contains an accident or a normal non-accident scene. It uses TensorFlow/Keras with MobileNetV2 transfer learning, trained on a custom image dataset.

## Features
- MobileNetV2 transfer learning with ImageNet weights
- Image size of 224x224
- Data augmentation: rotation, horizontal flip, zoom, brightness
- Categorical cross-entropy loss and Adam optimizer
- Training for 20 epochs with ModelCheckpoint and EarlyStopping
- Training and validation accuracy/loss plots
- Test set evaluation with accuracy, precision, recall, F1-score, confusion matrix, and classification report
- Single-image prediction script
- Real-time webcam detector with OpenCV
- Localhost web interface for image upload and prediction

## Dataset Structure
```text
data/
├── train/
│   ├── Accident/
│   └── Non Accident/
├── val/
│   ├── Accident/
│   └── Non Accident/
└── test/
    ├── Accident/
    └── Non Accident/
```

## Installation
1. Clone or open this project folder.
2. Create a Python virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.11+
- TensorFlow/Keras
- OpenCV
- Flask
- Matplotlib
- scikit-learn
- Pillow

## Training Instructions
Run the training script:

```bash
python src/train.py
```

This will:
- train the model for 20 epochs
- save the best model to models/best_model.keras
- generate plots in outputs/

## Testing Instructions
Evaluate the saved model on the test set:

```bash
python src/evaluate.py
```

## Single Image Prediction
```bash
python src/predict.py --image path/to/image.jpg
```

## Real-Time Detection
```bash
python src/realtime.py --source 0
```

## Localhost Web App
Start the web app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Example Outputs
- Training plot: outputs/training_history.png
- Confusion matrix: outputs/confusion_matrix.png
- Metrics report: outputs/classification_report.txt
- Best model: models/best_model.keras

## Future Improvements
- Add more diverse accident images
- Explore larger backbone models such as EfficientNet
- Add video-file inference support
- Deploy to a cloud platform or container
