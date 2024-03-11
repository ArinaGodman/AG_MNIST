import streamlit as st
import pandas as pd
import numpy as np
import cv2

from predicction import make_prediction
#from prediction import predict

# Function to display a big prediction number
def display_prediction_number(predicted_number):
    st.write(f'## Prediction: {predicted_number}')

# Main Streamlit app
st.title('Reading Numbers')
st.markdown('Model to read a number from your photo')

# File uploader
uploaded_file = st.file_uploader('Upload your photo', type=['jpg', 'jpeg', 'png'])

# Button to challenge the model
if st.button('Challenge Model'):
    if uploaded_file is not None:
        # Add your model prediction logic here
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        predicted_number = make_prediction(image)  # Replace with the actual predicted number

        # Display the prediction number
         # Display the original and processed images
        st.image([image], caption=['Original Image'], width=300)
        display_prediction_number(predicted_number)
    else:
        st.warning('Please upload a photo before challenging the model.')

