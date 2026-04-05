import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease App", page_icon="🌿")

st.title("🌿 Plant Disease Detection App (Demo Safe Version)")

st.write("⚠️ Streamlit Cloud safe version without TensorFlow errors")

class_names = ["Apple Scab", "Black Rot", "Cedar Rust", "Healthy"]

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    st.info("🧠 Model not loaded in cloud-safe mode")

    # Dummy prediction (safe for deployment)
    prediction = np.random.choice(class_names)
    confidence = np.random.randint(70, 99)

    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence}%")
