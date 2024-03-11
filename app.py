import streamlit as st
import pandas as pd
import numpy as np
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
        predicted_number = predict  # Replace with the actual predicted number

        # Display the prediction number
        display_prediction_number(predicted_number)
    else:
        st.warning('Please upload a photo before challenging the model.')

