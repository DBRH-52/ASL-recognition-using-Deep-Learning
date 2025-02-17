# app_esp.py
from flask import Flask, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('asl_model.h5') 

@app.route('/')
def home():
    return "Welcome to the ASL Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Receive image from ESP32Cam
    image_data = request.get_data()
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE) / 255.0
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0)

    # Perform prediction
    prediction = model.predict(image)

    # Get the predicted label
    predicted_label = chr(ord('A') + np.argmax(prediction))

    return predicted_label

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
