# DIP_Fruit-Freshness-Detection
# Fruit Freshness Detection using Image Processing and Deep Learning

This project uses **Digital Image Processing (DIP)** techniques and a **Convolutional Neural Network (CNN)** to detect the freshness of fruits and vegetables. It supports image inputs from both a dataset and a **live camera feed**, offering real-time predictions about whether a fruit is fresh or rotten.

---

## Features

- Image preprocessing using OpenCV (grayscale, edge detection, histogram equalization, etc.)
- CNN model trained on augmented dataset of fresh and rotten fruits
- Real-time freshness detection using a live camera
- Predicts both fruit type and its condition (Fresh/Rotten)
- Applications in retail, warehousing, agriculture, and vending systems

---

## Project Structure

Fruit-Freshness-Detection/ │ ├── dataset/ # Contains training and testing images ├── model/ # Saved CNN model files ├── src/ # Main source code │ ├── preprocess.py # Image preprocessing pipeline │ ├── train_model.py # CNN model training │ ├── predict.py # Inference using trained model │ ├── live_camera.py # Real-time prediction with camera │ ├── flowchart.png # System design flowchart ├── README.md # Project documentation └── requirements.txt # List of dependencies

yaml
Copy
Edit

---

## Technologies Used

- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy & Matplotlib
- Jupyter Notebook / Google Colab

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/BhumikaMunot/DIP_Fruit-Freshness-Detection.git
cd Fruit-Freshness-Detection
Create a virtual environment (optional but recommended)


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies


pip install -r requirements.txt
How to Use
Run Real-Time Detection
run: python app.py

Train the Model (if you want to retrain)
python src/train_model.py

Applications
Supermarkets for automatic fruit sorting

Cold storages and warehouses for real-time monitoring

Vending machines for quality control

Agricultural export grading systems

References
OpenCV Documentation

TensorFlow Documentation

Python Official Docs

Kaggle datasets for fruit freshness detection
