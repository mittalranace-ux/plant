import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Plant Disease Detection App")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# ---------- Classes ----------
class_names = ["Apple Scab", "Black Rot", "Cedar Rust", "Healthy"]

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]
    conf = np.max(prediction)*100

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {conf:.2f}%")
