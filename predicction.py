import joblib
import streamlit as st
import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    resized_img = cv2.resize(gray_img, (28, 28))

    # Apply thresholding
    _, threshold_img = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Normalize pixel values
    normalized_img = threshold_img / 255.0

    # Reshape to match the input shape of the MNIST model
    reshaped_img = normalized_img.reshape(28, 28, 1)

    return reshaped_img

def make_prediction(image):
    image_pp = preprocess_image(image)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'svc_model.sav')
    model = joblib.load(model_path)
    predicted_number = model.predict([image_pp.flatten()])  # Flatten the image array
    return predicted_number[0]  # Get the first element from the prediction array