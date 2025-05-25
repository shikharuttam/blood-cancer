import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from flask import Flask, render_template, request, send_file, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Suppress Werkzeug logs
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load the model
model = load_model('improved_blood_cell_model_v2.h5')

# Recompile the model to set up metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class names
class_names = ['Benign', '[Malignant] Pre-B', '[Malignant] Pro-B', '[Malignant] early Pre-B']

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('prediction_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT,
                  predicted_class TEXT,
                  confidence FLOAT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

def preprocess_image(image_path):
    """Preprocess the uploaded image for prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    """Make a prediction using the loaded model."""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence_scores = prediction
    return class_index, confidence_scores

def create_confidence_chart(confidence_scores):
    """Create a bar chart of confidence scores and save it."""
    try:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=confidence_scores, y=class_names, hue=class_names, palette='Blues_d', legend=False)
        plt.xlabel('Confidence Score')
        plt.title('Prediction Confidence Scores')
        plt.tight_layout()
        chart_path = os.path.join('static', 'prediction_chart.png')
        plt.savefig(chart_path)
        plt.close()
        return chart_path
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def image_to_base64(image_path):
    """Convert image to base64 string for display in HTML."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    if file:
        # Validate file type
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return 'Invalid file type. Please upload a JPEG or PNG image.', 400

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Validate that the file is a readable image
        try:
            img = Image.open(file_path)
            img.verify()
            img.close()
            img = Image.open(file_path)
        except Exception as e:
            return f'Error: Invalid or corrupted image file - {str(e)}', 400

        # Make prediction
        class_index, confidence_scores = predict_image(file_path)
        predicted_class = class_names[class_index]
        max_confidence = float(confidence_scores[class_index])
        confidence_scores = [float(score) for score in confidence_scores]

        # Create confidence chart
        chart_path = create_confidence_chart(confidence_scores)

        # Convert image to base64 for display
        image_base64 = image_to_base64(file_path)

        # Save prediction to database
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn = sqlite3.connect('prediction_history.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (image_path, predicted_class, confidence, timestamp) VALUES (?, ?, ?, ?)",
                      (file_path, predicted_class, max_confidence, timestamp))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

        # Prepare prediction text for download
        prediction_text = f"Predicted Class: {predicted_class}\nConfidence: {max_confidence:.2f}\nTimestamp: {timestamp}"
        with open('prediction.txt', 'w') as f:
            f.write(prediction_text)

        # Zip class_names and confidence_scores
        class_confidence_pairs = list(zip(class_names, confidence_scores))

        return render_template(
            'result.html',
            predicted_class=predicted_class,
            confidence=max_confidence,
            class_confidence_pairs=class_confidence_pairs,
            image_base64=image_base64,
            chart_path=chart_path,
            filename=file.filename
        )

@app.route('/history')
def history():
    """Display prediction history."""
    conn = sqlite3.connect('prediction_history.db')
    c = conn.cursor()
    c.execute("SELECT image_path, predicted_class, confidence, timestamp FROM predictions ORDER BY timestamp DESC")
    history = c.fetchall()
    conn.close()

    history_with_images = []
    for entry in history:
        image_path, predicted_class, confidence, timestamp = entry
        try:
            image_base64 = image_to_base64(image_path)
            history_with_images.append({
                'image_base64': image_base64,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'timestamp': timestamp
            })
        except:
            continue

    return render_template('history.html', history=history_with_images)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the prediction history."""
    conn = sqlite3.connect('prediction_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return redirect(url_for('history'))

@app.route('/download_prediction')
def download_prediction():
    """Download the prediction result as a text file."""
    return send_file('prediction.txt', as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)