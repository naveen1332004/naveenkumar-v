import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 

st.header('fashion recommendation system')
Image_features = pkl.load(open('Fashion Recommendation/Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
model.summary()
upload_file = st.file_uploader("Upload Image")