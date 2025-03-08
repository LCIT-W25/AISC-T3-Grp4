import pytest
import tensorflow as tf
import os
import requests

# Define paths to your models
gru_model_url = 'https://github.com/LCIT-W25/AISC-T3-Grp4/releases/download/vgg_rnn_gru/best_hyperparameter_gru_model.h5'
vgg_model_url = 'https://github.com/LCIT-W25/AISC-T3-Grp4/releases/download/vgg_rnn_gru/best_rnn_model.h5'
rnn_model_url = 'https://github.com/LCIT-W25/AISC-T3-Grp4/releases/download/vgg_rnn_gru/VGG16CLASSIFIER.h5'

# Define local paths to save the models
gru_model_path = 'models/gru_model.h5'
vgg_model_path = 'models/vgg_model.h5'
rnn_model_path = 'models/rnn_model.h5'

def download_model(url, path):
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

@pytest.fixture(scope="module")
def load_gru_model():
    if not os.path.exists(gru_model_path):
        download_model(gru_model_url, gru_model_path)
    return tf.keras.models.load_model(gru_model_path)

@pytest.fixture(scope="module")
def load_vgg_model():
    if not os.path.exists(vgg_model_path):
        download_model(vgg_model_url, vgg_model_path)
    return tf.keras.models.load_model(vgg_model_path)

@pytest.fixture(scope="module")
def load_rnn_model():
    if not os.path.exists(rnn_model_path):
        download_model(rnn_model_url, rnn_model_path)
    return tf.keras.models.load_model(rnn_model_path)

def test_gru_model(load_gru_model):
    model = load_gru_model
    assert model is not None, "GRU model failed to load"
    model.summary()

def test_vgg_model(load_vgg_model):
    model = load_vgg_model
    assert model is not None, "VGG model failed to load"
    model.summary()

def test_rnn_model(load_rnn_model):
    model = load_rnn_model
    assert model is not None, "RNN model failed to load"
    model.summary()

if __name__ == "__main__":
    pytest.main()
