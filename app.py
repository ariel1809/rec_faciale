import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title('Image classification')
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg","png"])
generated_pred = st.sidebar.button("Generate Prediction")
model = load_model('modelCNN.h5')
classes_p = {'Infection_Bacterienne': 0,'Infection_Covid': 1,'Infection_Virale': 2,'Normal': 3}
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    test_image = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = img_to_array(test_image)
    img_array = np.expand_dims(img_array, axis=0)

    if generated_pred is True:
        prediction = model.predict(img_array)
        classes = np.argmax(prediction[0])
        for key, value in classes_p.items():
            if value == classes:
                st.title(f'the result of the radiography is {key}')