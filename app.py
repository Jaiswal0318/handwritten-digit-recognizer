
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Digit AI | Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM DARK THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #080d1b 0%, #10172a 55%, #11102a 100%);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    padding: 32px;
    border: 1px solid #293653;
    border-radius: 24px;
    background: linear-gradient(120deg, #151f38, #17142f);
    margin-bottom: 28px;
}

.eyebrow {
    color: #60a5fa;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 800;
    letter-spacing: -1.8px;
    margin: 12px 0;
    color: #f8fafc;
}

.hero p {
    color: #aab8d0;
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    color: #f1f5f9;
    font-size: 21px;
    font-weight: 700;
    margin: 12px 0 18px;
}

.panel {
    background: rgba(20, 29, 50, 0.85);
    border: 1px solid #2a3856;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(145deg, #182541, #171b32);
    border: 1px solid #334466;
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;
}

.metric-label {
    color: #9aabc6;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 800;
}

.prediction-card {
    text-align: center;
    padding: 26px;
    border-radius: 20px;
    border: 1px solid #4657a8;
    background: linear-gradient(145deg, #17284a, #211b45);
}

.prediction-digit {
    font-size: 100px;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(90deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.muted {
    color: #9aabc6;
    font-size: 14px;
}

div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 13px 20px;
    color: white;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border: none;
    color: white;
    filter: brightness(1.15);
}

[data-testid="stFileUploader"] {
    background: #111a2e;
    border: 1px dashed #53658a;
    border-radius: 16px;
    padding: 16px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label {
    color: #b8c5dc !important;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

h2, h3, p, label {
    color: #e2e8f0;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293653;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #293653;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI • COMPUTER VISION • DEEP LEARNING</div>
    <h1>🔢 Digit AI Studio</h1>
    <p>
        Turn handwritten digits into predictions.
        Upload an image, run your trained neural network,
        and explore the model's confidence across digits 0–9.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINED MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/digit_recognizer.keras"
    )


try:
    model = load_model()
except Exception as error:
    st.error(
        "The trained model could not be loaded. "
        "Please check that models/digit_recognizer.keras exists."
    )
    st.stop()


# ---------------- MAIN DASHBOARD ----------------
left_column, right_column = st.columns([1, 1.15], gap="large")

with left_column:
    st.markdown(
        '<div class="section-title">01 / Upload your digit</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("**Choose an image**")
    st.caption("Supported formats: PNG, JPG, JPEG")

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("L")

            st.image(
                original_image,
                caption="Your uploaded image",
                use_container_width=True
            )

            st.caption(
                "For best results, use a clear image containing "
                "one centered handwritten digit."
            )

        except Exception:
            st.error("Unable to read this image. Please upload another.")
            uploaded_file = None

    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding:30px 8px;
            color:#93a4c2;
        ">
            <div style="font-size:48px;">✦</div>
            <div style="font-size:16px;font-weight:600;color:#dbeafe;">
                Your digit starts here
            </div>
            <p class="muted">
                Upload an image to begin recognition.
            </p>
        </div>
        """, unsafe_allow_html=True)

    recognize_clicked = st.button(
        "✦  Recognize Digit",
        disabled=uploaded_file is None
    )

    st.markdown('</div>', unsafe_allow_html=True)


with right_column:
    st.markdown(
        '<div class="section-title">02 / Recognition results</div>',
        unsafe_allow_html=True
    )

    if recognize_clicked and uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("L")

            # Match the existing model's preprocessing
            image = ImageOps.invert(original_image)
            image = ImageOps.fit(image, (28, 28))

            image_array = (
                np.array(image).astype("float32") / 255.0
            )
            image_array = image_array.reshape(1, 28, 28, 1)

            predictions = model.predict(image_array, verbose=0)[0]
            probabilities = predictions * 100

            digit = int(np.argmax(predictions))
            confidence = float(probabilities[digit])

            st.markdown(f"""
            <div class="prediction-card">
                <div class="eyebrow">PREDICTED DIGIT</div>
                <div class="prediction-digit">{digit}</div>
                <div style="color:#c4b5fd;font-size:15px;font-weight:600;">
                    Recognition complete
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            metric_one, metric_two = st.columns(2)

            with metric_one:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Top prediction</div>
                    <div class="metric-value">{digit}</div>
                </div>
                """, unsafe_allow_html=True)

            with metric_two:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model confidence</div>
                    <div class="metric-value">{confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown(
                '<div class="section-title">Confidence distribution</div>',
                unsafe_allow_html=True
            )

            chart_data = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Confidence (%)": probabilities
            }).set_index("Digit")

            st.bar_chart(
                chart_data,
                y="Confidence (%)",
                height=280,
                use_container_width=True
            )

            with st.expander("View all digit probabilities"):
                probability_table = pd.DataFrame({
                    "Digit": list(range(10)),
                    "Probability (%)": [
                        round(float(p), 2) for p in probabilities
                    ]
                }).sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.dataframe(
                    probability_table,
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as error:
            st.error(
                "Prediction failed. Please try another image."
            )
            st.exception(error)

    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:55px 20px;">
            <div style="font-size:52px;">◎</div>
            <h3 style="color:#e2e8f0;">Waiting for input</h3>
            <p class="muted">
                Your prediction and confidence analysis
                will appear here after recognition.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("---")

footer_left, footer_right = st.columns(2)

with footer_left:
    st.markdown(
        '<p class="muted">DIGIT AI STUDIO · Handwritten Digit Recognition</p>',
        unsafe_allow_html=True
    )

with footer_right:
    st.markdown(
        '<p class="muted" style="text-align:right;">'
        'Powered by TensorFlow · CNN · Streamlit'
        '</p>',
        unsafe_allow_html=True
    )
