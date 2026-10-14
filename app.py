
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit from 0 to 9.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/digit_recognizer.keras"
    )

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("L")
    st.image(
        original_image,
        caption="Uploaded Image",
        width=200
    )

    if st.button("Recognize Digit"):
        image = ImageOps.invert(original_image)
        image = ImageOps.fit(image, (28, 28))

        image_array = np.array(image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        predictions = model.predict(image_array, verbose=0)
        probabilities = predictions[0] * 100

        digit = int(np.argmax(predictions[0]))
        confidence = float(probabilities[digit])

        st.success(f"Predicted Digit: {digit}")
        st.write(f"Model confidence: {confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        chart_data = {
            "Digit": list(range(10)),
            "Confidence (%)": probabilities.tolist()
        }

        st.bar_chart(
            chart_data,
            x="Digit",
            y="Confidence (%)"
        )

        for i, probability in enumerate(probabilities):
            st.write(f"Digit {i}: {probability:.2f}%")
