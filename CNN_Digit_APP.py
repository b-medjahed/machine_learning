import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import keras
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas



## loading CNN model
model=keras.models.load_model(r'C:\Users\cmedj\OneDrive\Documents\Raf Course\week6\model_cnn')



## set page configuration
st.set_page_config(page_title='Digit Recogniser', layout='centered')

## add page title and content
st.title('Digit Recogniser Classification using Convolutional Neural Network')
st.write('Please scroll down and upload your image:')

st.title("Drawable Canvas")
st.markdown("""Draw your digit on the canvas below""")
st.sidebar.header("Configuration")

# Specify parameters and drawing mode
stk_witdth = st.sidebar.slider("Pen width: ", 1, 25, 19)
Drawing_Mode = st.sidebar.checkbox("Drawing mode ?", True)

# Create a canvas component
canvas_res = st_canvas(
    stroke_width=stk_witdth, stroke_color='#FFFFFF', background_color='#000000', fill_color='rgba(255, 165, 0, 0.3)', width=256, height=256,drawing_mode='freedraw', update_streamlit=Drawing_Mode, key="canvas",)



def process(img):
        
        img = img.resize((28, 28)).convert('L')
        sample_array=np.array(img)
        sample_array=sample_array.astype(np.float16)
        sample_array=sample_array/255
        sample_array = np.expand_dims(sample_array, axis=0)
        sample_array = np.expand_dims(sample_array, axis=-1)
        return(sample_array)





if canvas_res.image_data is not None:
    
    st.image(canvas_res.image_data)
    #st.write(type(canvas_res.image_data))
    #st.write(canvas_res.image_data.shape)
    #st.write(canvas_res.image_data)
    img = Image.fromarray(canvas_res.image_data.astype('uint8'), mode="RGBA")
    sample= process(img)
     

if st.button('Predict'):
        prediction= np.argmax(model.predict(sample))
        st.write('Predicted digit is:' , prediction)
    





