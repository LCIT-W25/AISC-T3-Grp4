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
from sklearn.exceptions import NotFittedError

# Custom CSS to fit content within the screen
st.markdown(
    """
    <style>
    .main {
        padding: 10px 20px;
        max-width: 1000px;
        margin: auto;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    .css-18e3th9 {
        padding: 1rem 2rem;
    }
    img {
        max-height: 300px;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
dnn_model = load_model('savedmodels/dnn_model.h5')                # DNN Model (Untuned)
resnet_model = load_model('savedmodels/resnet_model.h5')             # ResNet Model (Tuned)
untuned_nb_model = joblib.load('savedmodels/untunednb.pkl')       # First version NB
best_pipeline = joblib.load('savedmodels/best_pipeline.pkl')      # Tuned NB
svm_model = joblib.load('savedmodels/svm_model.pkl')              # SVM Model (Untuned)
vectorizer = joblib.load('savedmodels/vectorizer.pkl')            # Vectorizer for SVM
knn_model = joblib.load('savedmodels/knn_model.pkl')              # kNN Model for Image Classification
label_encoder = joblib.load('savedmodels/label_encoder.pkl')      # Label Encoder for kNN

# Label encoder classes for image classification
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

# Function to preprocess the uploaded image
def preprocess_image(img, model_type):
    if model_type == "DNN (Tuned)":
        img = img.resize((224, 224))  # ResNet input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # ResNet-specific preprocessing
    elif model_type == "DNN (Untuned)":
        img = img.resize((64, 64))  # Resize to match DNN input (64x64x3)
        img_array = image.img_to_array(img)
        img_array = img_array.flatten().reshape(1, -1)  # Flatten for DNN
    elif model_type == "kNN (UNTuned)":
        img = img.resize((32, 32))  # Resize to match kNN input (32x32x3 = 3072)
        img_array = image.img_to_array(img)
        img_array = img_array.flatten().reshape(1, -1)  # Flatten for kNN

    return img_array

# Streamlit App
st.sidebar.title("Select Task")
task = st.sidebar.radio("Choose a task:", ("Image Classification", "Sentiment Analysis"))

if task == "Image Classification":
    st.title("🖼️ Image Classifier")
    model_choice = st.selectbox("Choose Model:", ("DNN (Untuned)", "DNN (Tuned)", "kNN (UNTuned)"))

    model_map = {
        "DNN (Untuned)": dnn_model,
        "DNN (Tuned)": resnet_model,
        "kNN (UNTuned)": knn_model
    }

    selected_model = model_map[model_choice]

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        img_array = preprocess_image(img, model_choice)

        if st.button("Predict"):
            if model_choice == "kNN (Untuned)":
                prediction = selected_model.predict(img_array)[0]
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                st.success(f"✅ Predicted Class: **{predicted_class}**")
            else:
                predictions = selected_model.predict(img_array)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)

                st.success(f"✅ Predicted Class: **{class_labels[predicted_class]}**")
                st.info(f"🔍 Confidence: {confidence * 100:.2f}%")

elif task == "Sentiment Analysis":
    st.title("📝 Sentiment Analysis App")
    st.write("Enter a sentence below to predict its sentiment.")

    model_choice = st.selectbox("Choose Model:", ("Tuned Naive Bayes", "Untuned Naive Bayes", "Untuned SVM"))

    model_map = {
        "Tuned Naive Bayes": best_pipeline,
        "Untuned Naive Bayes": untuned_nb_model,
        "Untuned SVM": svm_model
    }

    selected_model = model_map[model_choice]

    user_input = st.text_area("Type your sentence here:", "")

    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    if st.button("Analyze Sentiment"):
        if user_input:
            try:
                if model_choice == "Untuned SVM":
                    user_input_transformed = vectorizer.transform([user_input])
                    prediction = selected_model.predict(user_input_transformed)[0]
                    probabilities = selected_model.predict_proba(user_input_transformed)[0]
                else:
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

            except NotFittedError:
                st.error("❌ The selected SVM model is not fitted. Please retrain the model and save it correctly.")
        else:
            st.warning("Please enter a sentence to analyze.")
