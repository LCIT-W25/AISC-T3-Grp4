import tensorflow as tf
import joblib
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import pytest

# Define paths using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent  # Get root project directory
MODELS_DIR = BASE_DIR / "models"  # Define models directory

# Load models
gru_model_path = MODELS_DIR / "best_hyperparameter_gru_model.h5"
rnn_model_path = MODELS_DIR / "best_rnn_model.keras"

if not gru_model_path.exists():
    raise FileNotFoundError(f"GRU Model not found: {gru_model_path}")

if not rnn_model_path.exists():
    raise FileNotFoundError(f"RNN Model not found: {rnn_model_path}")

gru_model = tf.keras.models.load_model(gru_model_path)
rnn_model = tf.keras.models.load_model(rnn_model_path)

# Load tokenizers
gru_tokenizer = joblib.load(MODELS_DIR / "tokenizer.joblib")
rnn_tokenizer = joblib.load(MODELS_DIR / "tokenizer_rnn.joblib")

# Load sentiment mappings
gru_sentiment_mapping = joblib.load(MODELS_DIR / "sentiment_mapping.pkl")
rnn_sentiment_mapping = joblib.load(MODELS_DIR / "sentiment_mapping_rnn.joblib")

@pytest.mark.parametrize("model, tokenizer, sentiment_mapping", [
    (gru_model, gru_tokenizer, gru_sentiment_mapping),
    (rnn_model, rnn_tokenizer, rnn_sentiment_mapping)
])
def test_model_prediction(model, tokenizer, sentiment_mapping):
    """Test if model predicts sentiment correctly."""
    test_sentence = "This is a wonderful experience!"
    sequence = tokenizer.texts_to_sequences([test_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    
    prediction_prob = model.predict(padded_sequence)
    sentiment_idx = np.argmax(prediction_prob, axis=1)[0]
    
    assert sentiment_idx in sentiment_mapping.values(), "Invalid model prediction index"

def test_empty_input():
    """Test if model handles empty input properly."""
    empty_text = ""
    sequence = gru_tokenizer.texts_to_sequences([empty_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    prediction_prob = gru_model.predict(padded_sequence)
    assert prediction_prob is not None, "Model should return a prediction even for empty input"

def test_non_string_input():
    """Test if tokenizer can handle non-string input."""
    non_string_input = 12345  # Integer instead of string
    with pytest.raises(AttributeError):
        sequence = gru_tokenizer.texts_to_sequences([non_string_input])

def test_out_of_vocab_words():
    """Test model behavior with unknown words."""
    test_sentence = "flibberflabberdoodle"  # Unlikely to be in the vocabulary
    sequence = gru_tokenizer.texts_to_sequences([test_sentence])
    
    assert len(sequence[0]) == 0, "Tokenizer should return an empty sequence for unknown words"

def test_model_prediction_time():
    """Test if model prediction is within acceptable limits (under 500ms)."""
    test_sentence = "I absolutely love this product!"
    sequence = gru_tokenizer.texts_to_sequences([test_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    start_time = time.time()
    _ = gru_model.predict(padded_sequence)
    end_time = time.time()

    assert (end_time - start_time) < 0.5, "Model inference took too long!"

def test_tokenizer_vocabulary_size():
    """Check if tokenizer vocabulary size is within expected range."""
    vocab_size = len(gru_tokenizer.word_index)
    assert vocab_size > 5000, "Tokenizer vocabulary size is too small!"
