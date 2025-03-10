import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define model path (make sure this file exists locally)
MODEL_PATH = "model/metal_model.tflite"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels - adjust as per your model training
class_names = ['Gold', 'Silver', 'Copper', 'Iron']

# Preprocess image for prediction
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)     # Add batch dimension
    return img

# Predict using the TFLite model
def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = float(np.max(output_data))
    return class_names[predicted_index], confidence

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Detect page
@app.route("/detect")
def detect():
    return render_template("index.html")

# Predict API endpoint (POST with image file)
@app.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    try:
        # Convert uploaded image to OpenCV format
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Make prediction
        label, confidence = predict(image)

        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app (bind to 0.0.0.0 for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
