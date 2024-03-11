import joblib
import streamlit as st
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(gray_image, (28, 28))

    # Reshape to match the input shape of the MNIST model
    reshaped_image = resized_image.reshape(28, 28)

    normalized_image = reshaped_image / 255.0

    inverted_image = (1 - normalized_image) * 255.0

    # Flatten to a 1D array
    flattened_img = inverted_image.flatten()

    # Reshape to a 2D array for StandardScaler
    reshaped_img_2d = flattened_img.reshape(-1, 1)

    current_dir = os.path.dirname(__file__)
    std_scaler_path = os.path.join(current_dir, 'std_scaler.sav')
    std_scaler = joblib.load(std_scaler_path)

    # Apply StandardScaler
    img_ready = std_scaler.transform(reshaped_image)

    return img_ready.flatten()

def make_prediction(image):
    image_pp = preprocess_image(image)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'svc_model.sav')
    model = joblib.load(model_path)
    predicted_number = model.predict([image_pp.flatten()])  # Flatten the image array
    return predicted_number[0]  # Get the first element from the prediction array