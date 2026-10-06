# ASL Sign Language Recognition (Computer Vision & CNN)

A real-time American Sign Language (ASL) alphabet recognition system designed with a strong focus on **computer vision and image processing**, utilizing a Convolutional Neural Network (CNN) as an auxiliary classification model.

## 🛠️ Technologies & Libraries
* **Python**
* **PyTorch** (Convolutional Neural Networks / CNNs)
* **OpenCV** (Image processing and HSV color filtering)
* **NumPy**

## 🚀 Project Pipeline
1. **Capture & Vision (OpenCV):** Webcam frame acquisition restricted to a Region of Interest (ROI), applying color segmentation in the **HSV** color space (skin tone) for hand detection and isolation.
2. **Preprocessing:** Resizing and normalization of incoming RGB images ($64\times64$), along with data augmentation techniques to enhance robustness.
3. **Inference (PyTorch):** Classification of 26 ASL alphabet classes using a CNN architecture (convolutional layers, ReLU activations, pooling, and Dropout), displaying real-time confidence probabilities.

## ⚙️ Repository Structure
* `dataset/` -> Images used for the model training
* `models/` -> Trained model weights (`.pth`).
* `src/` -> Source code (preprocessing, training, and real-time execution script).
