import streamlit as st
import requests
import json
from PIL import Image
import io

# ========================== SIDEBAR NAVIGATION ========================== #
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose an Application:", ["Sentiment Analysis", "Image Classification"])

# ========================== SENTIMENT ANALYSIS ========================== #
if app_mode == "Sentiment Analysis":
    st.title("🔍 Sentiment Analysis (GRU & RNN)")
    st.subheader("Enter a sentence and select a model.")

    # Model Selection
    model_choice = st.radio("Choose a Model:", ("GRU", "RNN"))

    # Text Input
    user_input = st.text_area("Enter your text:")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            api_url = "http://sentiment-api:5000/predict"  # Docker network alias
            data = {"text": user_input, "model": model_choice}
            response = requests.post(api_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Sentiment ({model_choice} Model):** {result['sentiment']}")
                st.write(f"**Confidence Scores:** {result['confidence']}")
            else:
                st.error("Error communicating with Sentiment API")
        else:
            st.warning("⚠️ Please enter text to analyze.")

# ========================== IMAGE CLASSIFICATION ========================== #
if app_mode == "Image Classification":
    st.title("🖼️ Image Classification (RNN & VGG16)")
    st.write("Upload an image to predict its class.")

    model_choice = st.selectbox("Select a model:", ["RNNCLASSIFIER", "VGG16CLASSIFIER"])

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Image Class"):
            api_url = "http://image-api:5001/predict"  # Docker network alias
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()
            
            files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
            data = {"model": model_choice}
            response = requests.post(api_url, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                st.success(f"**Predicted Class:** {result['predicted_class']}")
                st.write(f"**Confidence:** {result['confidence']:.2f}")
            else:
                st.error("Error communicating with Image API")
