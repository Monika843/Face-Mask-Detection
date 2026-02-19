# Face-Mask-Detection
This project detects whether a person is wearing a face mask or not using deep learning. It can be used for monitoring public safety in workplaces, hospitals, or public spaces.

Features

Detects masked and unmasked faces in real-time.

Supports image, video, and webcam input.

Built using Convolutional Neural Networks (CNNs) for high accuracy.

Demo

(Optional: Add a GIF or image showing the detection in action)

Dataset

Dataset consists of labeled images of people with masks and without masks.

Public datasets used:

Face Mask Dataset

[Additional images from custom collection]

Installation

Clone the repository:

git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt

Usage
1. Train the Model
python train.py


The model will be trained on the dataset and saved as mask_detector.model.

2. Real-time Detection
python detect_mask_video.py


Opens webcam and detects masked/unmasked faces in real-time.

3. Detect from Image
python detect_mask_image.py --image path/to/image.jpg

Requirements

Python 3.7+

TensorFlow / Keras

OpenCV

NumPy, imutils, scikit-learn

Model Architecture

CNN with multiple convolutional and max-pooling layers.

Softmax classifier for two classes: Mask and No Mask.

Trained with data augmentation to improve generalization.

Results

Accuracy on test set: ~95%

Real-time detection: ~15 FPS on CPU

Future Improvements

Add mask type classification (cloth, surgical, N95).

Optimize model with TensorFlow Lite for mobile deployment.

Improve detection under low-light conditions.

References

Face Mask Detection using Keras

Kaggle Face Mask Dataset
