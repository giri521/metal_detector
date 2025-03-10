from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'metal_model.h5')
MODEL_URL = "https://drive.google.com/uc?id=1_ApPcrk2NAcC6ddxjktaTYjVbtCBfj0A"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model (if needed for future use)
model = load_model(MODEL_PATH)

camera = cv2.VideoCapture(0)

# Define HSV ranges for each metal
def detect_metal_color(hsv_img):
    metal_ranges = {
        "Gold": ((20, 100, 100), (30, 255, 255)),
        "Copper": ((5, 100, 50), (20, 255, 200)),
        "Silver": ((0, 0, 180), (180, 60, 255)),
        "Iron_or_Steel": ((0, 0, 80), (180, 50, 180)),
    }

    metal_counts = {}

    for metal, (lower, upper) in metal_ranges.items():
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        count = cv2.countNonZero(mask)
        metal_counts[metal] = count

    detected = max(metal_counts, key=metal_counts.get)
    return detected if metal_counts[detected] > 500 else "Unknown"

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        resized = cv2.resize(frame, (300, 300))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        detected = detect_metal_color(hsv)

        cv2.putText(frame, f"Detected: {detected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/detect', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (300, 300))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            prediction = detect_metal_color(hsv)

    return render_template('index.html', prediction=prediction, filename=filename)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
