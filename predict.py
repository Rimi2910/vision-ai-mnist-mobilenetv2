import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (96, 96)  # your image size
MODEL_DIR = "models"  # folder containing your .keras models
MODEL_NAMES = {
    "tl": "mnist_mobilenetv2_tl.keras",         # transfer learning model
    "finetuned": "mnist_mobilenetv2_finetuned.keras"  # fine-tuned model
}

# =========================
# LOAD MODEL
# =========================
def load_model(version="tl"):
    if version not in MODEL_NAMES:
        raise ValueError(f"Invalid version '{version}'. Choose from: {list(MODEL_NAMES.keys())}")
    model_path = os.path.join(MODEL_DIR, MODEL_NAMES[version])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)

# =========================
# PREDICT FUNCTION
# =========================
def predict(model, img_path):
    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 96, 96, 1)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return predicted_class, confidence

# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_version> <image_path>")
        print("Example: python predict.py finetuned sample_digit.png")
        sys.exit(1)

    model_version = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)

    model = load_model(model_version)
    predicted_digit, conf = predict(model, image_path)

    print(f"Predicted Digit: {predicted_digit} | Confidence: {conf:.2f}")
