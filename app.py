import streamlit as st
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
untuned_nb_model = joblib.load('savedmodels/untunednb.pkl')           # First version NB
best_pipeline = joblib.load('savedmodels/best_pipeline.pkl')          # Tuned NB
resnet_model = load_model('savedmodels/resnet_model.h5')             # ResNet Model

# Label encoder classes for image classification
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # ResNet input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ResNet-specific preprocessing
    return img_array

# Streamlit App
st.sidebar.title("Select Task")
task = st.sidebar.radio("Choose a task:", ("Image Classification", "Sentiment Analysis"))

if task == "Image Classification":
    st.title(" DNN Image Classifier")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        img_array = preprocess_image(img)

        if st.button("Predict"):
            predictions = resnet_model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            st.success(f"✅ Predicted Class: **{class_labels[predicted_class]}**")
            st.info(f"🔍 Confidence: {confidence * 100:.2f}%")

elif task == "Sentiment Analysis":
    st.title("📝 Sentiment Analysis App")
    st.write("Enter a sentence below to predict its sentiment.")

    model_choice = st.selectbox("Choose Model:", ("Tuned Naive Bayes", "Untuned Naive Bayes"))

    model_map = {
        "Tuned Naive Bayes": best_pipeline,
        "Untuned Naive Bayes": untuned_nb_model
    }

    selected_model = model_map[model_choice]

    user_input = st.text_area("Type your sentence here:", "")

    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    if st.button("Analyze Sentiment"):
        if user_input:
            prediction = selected_model.predict([user_input])[0]
            probabilities = selected_model.predict_proba([user_input])[0]

            st.subheader("Prediction:")
            st.write(f"**Sentiment:** {sentiment_labels[prediction]}")

            st.subheader("Confidence:")
            st.write(f"Positive: {probabilities[2]:.2f}, Neutral: {probabilities[1]:.2f}, Negative: {probabilities[0]:.2f}")

            st.subheader("LIME Explanation:")
            explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
            explanation = explainer.explain_instance(user_input, selected_model.predict_proba, num_features=5)
            st.pyplot(explanation.as_pyplot_figure())
        else:
            st.warning("Please enter a sentence to analyze.")
