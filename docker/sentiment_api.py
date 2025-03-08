from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and tokenizers
gru_model = tf.keras.models.load_model("../models/best_hyperparameter_gru_model.h5")
rnn_model = tf.keras.models.load_model("../models/best_rnn_model.keras")

gru_tokenizer = joblib.load("../models/tokenizer.joblib")
rnn_tokenizer = joblib.load("../models/tokenizer_rnn.joblib")

sentiment_mapping = joblib.load("../models/sentiment_mapping.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    model_choice = data.get("model", "GRU")

    model = gru_model if model_choice == "GRU" else rnn_model
    tokenizer = gru_tokenizer if model_choice == "GRU" else rnn_tokenizer

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding="post")

    prediction_prob = model.predict(padded_sequence)
    sentiment_idx = np.argmax(prediction_prob, axis=1)[0]
    sentiment_label = sentiment_mapping.get(sentiment_idx, "Unknown")

    return jsonify({"sentiment": sentiment_label, "confidence": prediction_prob.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
