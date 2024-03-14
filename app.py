import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

from predicction import make_prediction


def display_prediction_number(predicted_number):
    st.write(f'<h2 style="color:#333333;">Prediction: {predicted_number[0]}</h2>', unsafe_allow_html=True)

st.title('Arinas ML-model that can read your numbers!')
st.write('<p style="font-size:20px; color:#333333;">Here you can upload or draw a number and challenge my model to predict what number it is seeing.</p>', unsafe_allow_html=True)
st.write('<p style="font-size:20px; color:#333333;">Numbers should be drawn in the middle on the canvas/your picture with a black thick pen.</p>', unsafe_allow_html=True)
st.write('<p style="font-size:20px; color:#333333;">Remember that it is a simple Machine Learing algorithm, so, do not expect magic from it , hehe</p>', unsafe_allow_html=True)

drawing_mode = st.checkbox('Enable Drawing')

SIZE = 200

col1, col2 = st.columns(2)
if drawing_mode:
    canvas_result = st_canvas(
        stroke_width=20,  
        stroke_color="white",  
        background_color="black",  
        update_streamlit=True,  
        height=SIZE, 
        width=SIZE,  
        drawing_mode="freedraw",  
    )

    
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)

    if st.button('Predict'):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prediction = make_prediction(gray_image)
        display_prediction_number(prediction)

else:

    uploaded_file = st.file_uploader('Upload your photo', type=['jpg', 'jpeg', 'png'])

    
    if st.button('Predict'):
        if uploaded_file is not None:
            
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (28, 28))
            normalized_image = resized_image / 255.0
            inverted_image = (1 - normalized_image) * 255.0
            mean_pixel_value = np.mean(inverted_image)
            inverted_image[inverted_image <= mean_pixel_value] = 0  

            
            prediction = make_prediction(inverted_image)
            st.image(image, caption='Original Image', use_column_width=True)
            
            digit = prediction[0]
            display_prediction_number(prediction)

        else:
            st.warning('Please upload a photo before challenging the model.')