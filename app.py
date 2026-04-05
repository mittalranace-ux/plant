import streamlit as st
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf
import os

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image for prediction")

# ---------- Download model safely ----------
MODEL_PATH = "model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "YOUR_GOOGLE_DRIVE_LINK"  # 👈 replace this
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------- Classes ----------
class_names = ["Apple Scab", "Black Rot", "Cedar Rust", "Healthy"]

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
