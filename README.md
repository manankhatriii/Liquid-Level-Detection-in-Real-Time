# Liquid-Level-Detection-in-Real-Time

Pepsi Level Detection Using Deep Learning
This repository contains the code for a deep learning project aimed at detecting the levels of liquid (Full, Half, Low, Absent) in a Pepsi bottle using image data and pretrained models like DenseNet121. The model is trained to classify images into four categories of liquid levels, providing real-time feedback through webcam input.

Table of Contents
Project Overview
Technologies Used
Dataset Preparation
Model Architecture
Training
Real-Time Detection
Installation
Usage
Results
Future Work
Contributing
License
Project Overview
The goal of this project is to develop a machine learning model that can detect the levels of liquid in a Pepsi bottle from images. The project uses various deep learning architectures like DenseNet, MobileNet, and custom models to classify images into four categories:

Full
Half
Low
Absent
The final system can also be used for real-time detection through a webcam interface.

Technologies Used
Python 3.x
OpenCV
TensorFlow and TensorFlow Hub
Keras
Pre-trained models (DenseNet121, MobileNet, etc.)
Scikit-learn
NumPy
ImageDataGenerator for data augmentation
Webcam or IP camera for real-time inference
Dataset Preparation
The dataset consists of images labeled according to the liquid levels:

Full: Images of the bottle when it's full.
Half: Images of the bottle when it's half full.
Low: Images of the bottle when it's at a low level.
Absent: Images when the bottle is absent or empty.
Ensure that the dataset is structured in the following folders:

mathematica
Copy code
/Full
/Half
/Low
/Absent
Each folder contains the respective images. All images are resized to 224x224 pixels.

Model Architecture
We use DenseNet121 as the primary model for transfer learning. The architecture is as follows:

Base Model: DenseNet121 (pre-trained on ImageNet, used without the top layers).
Custom Layers:
GlobalAveragePooling2D
Dense layer with 128 units and ReLU activation
Dropout layer for regularization (rate = 0.5)
Final Dense layer with 4 output units and softmax activation (for 4 classes).
Other models also used:
MobileNetV2
ResNet50
VGG16
InceptionV3
Custom CNN models
Training
The model is trained using the following settings:

Loss function: Sparse categorical crossentropy
Optimizer: Adam with a learning rate of 0.0001
Data Augmentation: Applied using ImageDataGenerator for better generalization.
Batch size: 32
Epochs: 5 (can be increased based on performance)
To train the model:

bash
Copy code
python train_model.py
Real-Time Detection
The trained model can be used to perform real-time inference via webcam or IP camera. The camera feed is captured, preprocessed, and passed through the model to predict the liquid level in the Pepsi bottle.
