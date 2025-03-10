from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Path to the TFLite model inside the 'static' folder
MODEL_PATH = "static/model/metal_model.tflite"

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the metal labels (order must match training labels)
LABELS = ["Gold", "Silver", "Copper", "Iron/Steel"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/detect")
def detect():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"})

    # Read and preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction results
    prediction = np.squeeze(output_data)
    predicted_label = LABELS[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "label": predicted_label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
