import joblib
import streamlit as st
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_image(image, std_scaler_path='std_scaler.sav'):
    flattened_img = image.flatten()
    reshaped_img_2d = flattened_img.reshape(1, -1)  

   
    std_scaler = joblib.load(std_scaler_path)
    img_ready = std_scaler.transform(reshaped_img_2d)

    return img_ready.flatten()

def make_prediction(image):
    image_pp = preprocess_image(image)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'svc_model.sav')
    model = joblib.load(model_path)
    
    predicted_number = model.predict([image_pp])
    
    return predicted_number[0] 
