import joblib
import streamlit as st
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_image(image, std_scaler_path='std_scaler.sav'):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(gray_image, (28, 28))

    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0

    inverted_image = (1 - normalized_image) * 255.0

    # Flatten to a 1D array
    flattened_img = inverted_image.flatten()

    # Reshape to a 2D array for StandardScaler
    reshaped_img_2d = flattened_img.reshape(1, -1)  # Change the shape to (1, 784)

    # Load StandardScaler
    std_scaler = joblib.load(std_scaler_path)

    # Apply StandardScaler without considering feature names
    img_ready = StandardScaler().fit_transform(reshaped_img_2d)

    return img_ready.flatten()

def make_prediction(image):
    image_pp = preprocess_image(image)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'svc_model.sav')
    model = joblib.load(model_path)
    
    # Flatten the image array if needed (depends on the model input)
    # predicted_number = model.predict([image_pp.flatten()])
    predicted_number = model.predict([image_pp])
    
    return predicted_number[0]  # Get the first element from the prediction array
