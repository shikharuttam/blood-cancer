import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, send_file, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import csv

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

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence_scores = prediction
    return class_index, confidence_scores

def create_confidence_chart(confidence_scores):
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
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    if file:
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return 'Invalid file type. Please upload a JPEG or PNG image.', 400

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        try:
            img = Image.open(file_path)
            img.verify()
            img.close()
            img = Image.open(file_path)
        except Exception as e:
            return f'Error: Invalid or corrupted image file - {str(e)}', 400

        class_index, confidence_scores = predict_image(file_path)
        predicted_class = class_names[class_index]
        max_confidence = float(confidence_scores[class_index])
        confidence_scores = [float(score) for score in confidence_scores]

        chart_path = create_confidence_chart(confidence_scores)

        image_base64 = image_to_base64(file_path)

        # Disabled SQLite for production
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prediction_text = f"Predicted Class: {predicted_class}\nConfidence: {max_confidence:.2f}\nTimestamp: {timestamp}"
        with open('prediction.txt', 'w') as f:
            f.write(prediction_text)

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
    # Disabled SQLite for production
    history_with_images = []
    return render_template('history.html', history=history_with_images)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    # Disabled for production
    return redirect(url_for('history'))

@app.route('/download_prediction')
def download_prediction():
    return send_file('prediction.txt', as_attachment=True)

@app.route('/export_history', methods=['GET'])
def export_history():
    # Disabled for production
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Image Path', 'Predicted Class', 'Confidence', 'Timestamp'])
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='prediction_history.csv'
    )

@app.route('/model_info')
def model_info():
    model_info = {
        'architecture': 'EfficientNetB0',
        'input_shape': '(150, 150, 3)',
        'classes': class_names,
        'training_data': 'BloodCellCancerDataset',
        'accuracy': '92.97% (on test set)',
        'last_updated': 'May 2025'
    }
    return render_template('model_info.html', model_info=model_info)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
