from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import secrets

# Initializing the Flask application
app = Flask(__name__)

secret_key = secrets.token_hex(16)
app.secret_key = secret_key

ROOT_DIR = os.path.dirname(__file__) #Define root directory

# Define the upload folder
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and class indices during application startup
model = load_model(os.path.join(ROOT_DIR, "model", "xception_plant_disease_detection(25-02-2024).h5"))
with open(os.path.join(ROOT_DIR, "files", "class_indices_xception copy.json"), 'r') as f:
    class_indices = json.load(f)
with open(os.path.join(ROOT_DIR, "files", "remedies.json"), 'r') as f:
        remedies = json.load(f)

# Function to predict disease from an image
def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    return predicted_class_index, confidence

# Route for home page
@app.route('/')
def home():
    return render_template('hometemp.html')
 
# Route for uploading image
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('result', filename=filename))
    return render_template('uploadtemp.html')

# Route for displaying result
@app.route('/resulttemp/<filename>')
def result(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predicted_class_index, confidence = predict_disease(image_path)
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    # Extract plant name and disease name from class_indices
    predicted_plant_name = class_indices[str(predicted_class_index)].get('plant_name', "Unknown")
    predicted_disease_name = class_indices[str(predicted_class_index)].get('disease_name', "Unknown")
    predicted_class_name = class_indices[str(predicted_class_index)].get('class_name', "Unknown")
    
    return render_template('resulttemp.html', filename=filename, predicted_class_name=predicted_class_name,
                           predicted_plant_name=predicted_plant_name, predicted_disease_name=predicted_disease_name,
                           confidence=confidence)

# Route for displaying remedies
@app.route('/remediestemp/<disease>/<plantname>/<diseasename>')
def show_remedies(disease, plantname, diseasename):
    remedies_for_disease = remedies.get(disease, [])
    # Pass plant name and disease name to the template
    return render_template('remediestemp.html', disease=disease, predicted_plant_name=plantname, predicted_disease_name=diseasename, remedies=remedies_for_disease)
if __name__ == '__main__':
    app.run(debug=True)
