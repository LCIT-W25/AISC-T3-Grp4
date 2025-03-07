import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.preprocessing import image
from PIL import Image
from pathlib import Path
import pytest

# Define paths using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent  # Get root project directory
MODELS_DIR = BASE_DIR / "models"  # Define models directory

# Load models with path validation
mobilenet_path = MODELS_DIR / "MobileNetV2.h5"
vgg16_path = MODELS_DIR / "VGG16_model.h5"

if not mobilenet_path.exists():
    raise FileNotFoundError(f"MobileNetV2 Model not found: {mobilenet_path}")

if not vgg16_path.exists():
    raise FileNotFoundError(f"VGG16 Model not found: {vgg16_path}")

mobilenet_model = tf.keras.models.load_model(mobilenet_path)
vgg16_model = tf.keras.models.load_model(vgg16_path)

# Define index-to-label mapping
index_to_label = {
    0: "Food",
    1: "Drink",
    2: "Inside",
    3: "Outside",
    4: "Menu",
    5: "Other"
}

#  Function to create a blank test image
def preprocess_test_image():
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@pytest.mark.parametrize("model", [mobilenet_model, vgg16_model])
def test_model_prediction(model):
    """Test if models return valid class indices."""
    img_array = preprocess_test_image()
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]

    assert predicted_class_idx in index_to_label, "Invalid prediction index"

def test_invalid_image_format():
    """Test model behavior with incorrect image format."""
    invalid_image = np.random.rand(100, 100, 3)  # Incorrect shape
    with pytest.raises(ValueError):
        mobilenet_model.predict(np.expand_dims(invalid_image, axis=0))

def test_model_inference_time():
    """Test if model inference time is within acceptable limits (under 500ms)."""
    img_array = preprocess_test_image()

    start_time = time.time()
    _ = mobilenet_model.predict(img_array)
    end_time = time.time()

    assert (end_time - start_time) < 0.5, "Model inference took too long!"

def test_model_loading():
    """Check if models load without errors."""
    assert mobilenet_model is not None, "Failed to load MobileNetV2 model"
    assert vgg16_model is not None, "Failed to load VGG16 model"

def test_class_label_mapping():
    """Ensure all predicted classes map to valid labels."""
    img_array = preprocess_test_image()
    prediction = mobilenet_model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]

    assert predicted_class_idx in index_to_label, "Class label mapping issue!"
