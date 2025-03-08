import pytest
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_gru_model():
    print("\nRunning GRU Model Test...")
    model = load_model("models/gru_model.h5")
    assert model is not None, "GRU model failed to load!"
    assert isinstance(model, tf.keras.Model), "GRU model is not a valid Keras model!"
