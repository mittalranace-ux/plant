import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to detect plant disease using Deep Learning")

# ---------- MODEL PATH ----------
MODEL_PATH = "model.h5"

# ---------- SAFE MODEL LOADER ----------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model first time... Please wait")

        # 🔴 IMPORTANT: Replace with your Google Drive file link
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"

        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------- CLASS LABELS ----------
class_names = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy"
]

# ---------- UPLOAD IMAGE ----------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------- PREPROCESS ----------
    img = image.resize((224, 224))
    img = np.array(img)

    # remove alpha channel if exists
    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ---------- PREDICTION ----------
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # ---------- OUTPUT ----------
    st.success(f"🌱 Predicted Disease: {predicted_class}")
    st.info(f"🔬 Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + TensorFlow")
