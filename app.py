import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
import base64
st.header(":black[Flower Classification Model]")
flower_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

model = load_model(
    '/Users/mbhashyam/Desktop/Project/Flower_recog_Model/Flower_Recog_Model.h5')


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    Returns
    -------
    The background.
    '''
    main_bg_ext = "bg.jpg"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


side_bg = 'bg4.avif'
set_bg_hack(side_bg)


def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    flower_name = flower_names[np.argmax(result)]
    score_obtained = str(np.max(result) * 100)

    # Create outcome with different colors
    outcome = (
        f"<span style='color: black;font-size: 20px'>The image belongs to </span>"
        f"<span style='color: DeepPink;font-size: 24px'>{flower_name}</span>"
        f"<span style='color: black;font-size: 20px'> with a score of : </span>"
        f"<span style='color: green;font-size: 24px'>{score_obtained}</span>"
    )

    return outcome


uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, width=200)

    # st.markdown(classify_images(uploaded_file))

    outcome = classify_images(uploaded_file)
    st.markdown(outcome, unsafe_allow_html=True)
