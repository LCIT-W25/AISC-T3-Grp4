import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ========================== SIDEBAR NAVIGATION ========================== #
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose an Application:", ["Sentiment Analysis", "Image Classification"])

# ========================== SENTIMENT ANALYSIS ========================== #
if app_mode == "Sentiment Analysis":
    st.title("🔍 Sentiment Analysis with GRU & RNN Models")
    st.subheader("Enter a sentence, select a model, and get the predicted sentiment!")

    # Model Selection
    model_choice = st.radio("Choose a Sentiment Analysis Model:", ("GRU", "RNN"))

    #  Load GRU & RNN Models
    @st.cache_resource
    def load_gru_model():
        path = os.path.abspath("../models/best_hyperparameter_gru_model.h5")
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}")
            return None
        return load_model(path)

    @st.cache_resource
    def load_rnn_model():
        path = os.path.abspath("../models/best_rnn_model.keras")
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}")
            return None
        return load_model(path)

    model = load_gru_model() if model_choice == "GRU" else load_rnn_model()

    # Load Tokenizers
    @st.cache_resource
    def load_gru_tokenizer():
        path = "../models/tokenizer.joblib"
        if not os.path.exists(path):
            st.error(f"Tokenizer file not found: {path}")
            return None
        return joblib.load(path)

    @st.cache_resource
    def load_rnn_tokenizer():
        path = "../models/tokenizer_rnn.joblib"
        if not os.path.exists(path):
            st.error(f"Tokenizer file not found: {path}")
            return None
        return joblib.load(path)

    tokenizer = load_gru_tokenizer() if model_choice == "GRU" else load_rnn_tokenizer()

    # Load Sentiment Mappings
    @st.cache_resource
    def load_gru_sentiment_mapping():
        path = "../models/sentiment_mapping.pkl"
        if not os.path.exists(path):
            st.error(f"Sentiment mapping file not found: {path}")
            return None
        return joblib.load(path)

    @st.cache_resource
    def load_rnn_sentiment_mapping():
        path = "../models/sentiment_mapping_rnn.joblib"
        if not os.path.exists(path):
            st.error(f"Sentiment mapping file not found: {path}")
            return None
        return joblib.load(path)

    sentiment_mapping = load_gru_sentiment_mapping() if model_choice == "GRU" else load_rnn_sentiment_mapping()
    sentiment_labels = {v: k for k, v in sentiment_mapping.items()} if sentiment_mapping else {}

    # Text Preprocessing Function
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r"[^\w\s]", '', text)
        text = re.sub(r"\d+", '', text)
        return text.strip()

    # Text Input
    user_input = st.text_area("Enter your text:", "")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            cleaned_text = preprocess_text(user_input)
            sequence = tokenizer.texts_to_sequences([cleaned_text]) if tokenizer else None
            padded_sequence = pad_sequences(sequence, maxlen=100, padding='post') if sequence else None

            if model and padded_sequence is not None:
                prediction_prob = model.predict(padded_sequence)
                sentiment_idx = np.argmax(prediction_prob, axis=1)[0]
                sentiment_label = sentiment_labels.get(sentiment_idx, "Unknown")
                
                st.success(f"**Predicted Sentiment ({model_choice} Model):** {sentiment_label}")
                st.write(f"**Confidence Scores:** {prediction_prob.tolist()}")
            else:
                st.error("Error: Model or tokenizer is not loaded properly.")
        else:
            st.warning(" Please enter text to analyze.")

# ========================== IMAGE CLASSIFICATION ========================== #
if app_mode == "Image Classification":
    st.title(" Image Classification with RNN & VGG16(Transfer Learning)")
    st.write("Upload an image to predict its class.")

    model_choice = st.selectbox("Select a model:", ["MobileNetV2", "VGG16"])

    # Load Image Classification Models
    @st.cache_resource
    def load_image_model(model_name):
        path = f"../models/{model_name}.h5"
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}")
            return None
        return tf.keras.models.load_model(path)

    model = load_image_model(model_choice)

    #  Class Labels
    index_to_label = {
        0: "Food",
        1: "Drink",
        2: "Inside",
        3: "Outside",
        4: "Menu",
        5: "Other"
    }

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    #  Image Preprocessing Function
    def preprocess_image(img):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(img)

        if model:
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_class_name = index_to_label.get(predicted_class_idx, "Unknown")
            confidence = np.max(prediction)
            
            st.write(f"**Model Used:** {model_choice}")
            st.write(f"**Predicted Class:** {predicted_class_name} (Class {predicted_class_idx})")
            st.write(f"**Confidence:** {confidence:.2f}")
        else:
            st.error("Error: Model is not loaded properly.")
