from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import tensorflow as tf
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and class indices
model = tf.keras.models.load_model("multi_class_model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hist_eq = cv2.equalizeHist(gray)

    # Save processed images
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    paths = {}
    paths['original'] = image_path
    paths['gray'] = save_image('gray', gray, filename)
    paths['canny'] = save_image('canny', canny, filename)
    paths['blur'] = save_image('blur', blur, filename)
    paths['hist_eq'] = save_image('hist_eq', hist_eq, filename)

    # Add prediction
    prediction_label, prediction_confidence = predict_image(image_path)
    paths['prediction_label'] = prediction_label
    paths['prediction_confidence'] = prediction_confidence

    return paths

def save_image(tag, image, base_name):
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_{tag}.jpg")
    cv2.imwrite(path, image)
    return path

def predict_image(image_path):
    # Preprocess image for model prediction
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize image
    img_input = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_input)[0]
    label = class_names[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(path)

        result_paths = process_image(path)
        return render_template('result.html', paths=result_paths, source="Uploaded Image")
    return redirect(url_for('index'))

@app.route('/capture', methods=['POST'])
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        filename = f"{uuid.uuid4()}_capture.jpg"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(path, frame)

        result_paths = process_image(path)
        return render_template('result.html', paths=result_paths, source="Captured Image")
    else:
        return "Failed to capture image from camera"

if __name__ == '__main__':
    app.run(debug=True)
