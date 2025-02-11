import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import datetime  # Import datetime module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
class Config:
    # Get configuration from environment variables with defaults
    MODEL_PATH = os.environ.get('MODEL_PATH', './Model/pneumonia.h5')
    UPLOAD_FOLDER = '/tmp/uploads'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 10000))
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'

# Ensure upload directory exists
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)

# Load model
try:
    model = load_model(Config.MODEL_PATH)
    class_labels = ["Normal", "Pneumonia"]
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Pneumonia Detection API is active",
        "version": "1.0.0"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/predict/pneumonia', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = int(np.round(predictions[0][0]))
            predicted_label = class_labels[predicted_class]
            
            os.remove(file_path)
            return jsonify({
                "prediction": predicted_label,
                "confidence": float(predictions[0][0]),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Processing failed"}), 500

# Development server configuration
if __name__ == '__main__':
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )