import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Initialize model as None
model = None

def load_ml_model():
    global model
    model_path = './model/pneumonia.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)

# Load model on startup
try:
    model = load_ml_model()
except Exception as e:
    print(f"Error loading model: {str(e)}")

class_labels = ["Normal", "Pneumonia"]

if not os.path.exists('uploads'):
    os.makedirs('uploads')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/', methods=['GET'])
def working():
    if model is None:
        return "Warning: Model not loaded", 503
    return "Pneumonia Detection API is active"

@app.route('/predict/pneumonia', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        try:
            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = int(np.round(predictions[0][0]))
            predicted_label = class_labels[predicted_class]

            return jsonify({
                "prediction": predicted_label,
                "confidence": float(predictions[0][0])
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)