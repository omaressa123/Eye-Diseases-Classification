import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="Eye Disease Prediction",
    page_icon="👁️",
    layout="centered",
)

st.title("👁️ Eye Disease Prediction")
st.write("Upload an eye image to predict the presence of disease using a trained deep learning model.")

@st.cache_resource(show_spinner=False)
def load_eye_model():
    return load_model('eye_model_final.h5')

model = load_eye_model()

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    pred_class = np.argmax(prediction, axis=1)[0] if prediction.shape[-1] > 1 else int(prediction[0][0] > 0.5)
    st.success(f"Prediction: **{pred_class}**")
    st.write(f"Raw model output: {prediction.tolist()}")

st.markdown("---")
st.caption("This is a simplified demo. The model expects images of eyes and outputs a prediction.")