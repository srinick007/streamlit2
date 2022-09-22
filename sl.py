import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
st.write("Hello There")
mn=load_model("DeepVisionModel.h5")
mn.summary()


from PIL import Image
import numpy as np

import cv2
import time

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
while run:
    _, frame = camera.read()
    image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image1_b_w = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1_b_w=cv2.resize(image1_b_w,(256,256))

    time.sleep(0.01)
    
    _, frame1 = camera.read()
    image2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    image2_b_w = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2_b_w=cv2.resize(image2_b_w,(256,256))
    absdiff = cv2.absdiff(image1_b_w,image2_b_w)
    absdiff=np.dstack([absdiff]*3)
    absdiff1 = np.expand_dims(absdiff, axis = 0)
    pred=mn.predict(absdiff1)
    if pred==1:
        absdiff = cv2.putText(absdiff, 'signed', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW.image(absdiff)
        
    else:
        absdiff = cv2.putText(absdiff, 'Unsigned', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW.image(absdiff)
    
    

else:
    st.write('Stopped')
