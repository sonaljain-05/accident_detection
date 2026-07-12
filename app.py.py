!pip install streamlit
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

# -----------------------------
# Build Model
# -----------------------------
vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(150, 150, 3)
)

vgg_base.trainable = False

model = Sequential([
    vgg_base,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
])

# -----------------------------
# Load Classifier Weights
# -----------------------------
weights = np.load("classifier_head_weights.npz")

model.layers[2].set_weights([
    weights["dense_256_kernel"],
    weights["dense_256_bias"]
])

model.layers[3].set_weights([
    weights["output_kernel"],
    weights["output_bias"]
])

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to predict whether it is Healthy or Diseased.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        processed = preprocess(image)

        prediction = model.predict(processed)

        probability = prediction[0][0]

        if probability > 0.5:
            st.error("🌱 Prediction: Diseased")
            st.write(f"Confidence: {probability*100:.2f}%")
        else:
            st.success("🍃 Prediction: Healthy")
            st.write(f"Confidence: {(1-probability)*100:.2f}%")