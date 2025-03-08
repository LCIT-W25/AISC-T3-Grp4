from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os

# Get absolute path to the models directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")

# Load models with absolute paths
rnn_classifier = tf.keras.models.load_model(os.path.join(MODELS_DIR, "RNNCLASSIFIER.h5"))
vgg16_classifier = tf.keras.models.load_model(os.path.join(MODELS_DIR, "VGG16CLASSIFIER.h5"))

# Class labels
index_to_label = {
    0: "Food",
    1: "Drink",
    2: "Inside",
    3: "Outside",
    4: "Menu",
    5: "Other"
}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    model_choice = request.form.get("model", "RNNCLASSIFIER")

    model = rnn_classifier if model_choice == "RNNCLASSIFIER" else vgg16_classifier

    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class_name = index_to_label.get(predicted_class_idx, "Unknown")

    return jsonify({"predicted_class": predicted_class_name, "confidence": float(np.max(prediction))})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
