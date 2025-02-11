import os
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('./Model/pneumonia.h5')
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
    return "Pneumonia Detection API is active"

@app.route('/predict/pneumonia', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
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
            
            os.remove(file_path)
            return jsonify({
                "prediction": predicted_label,
                "confidence": float(predictions[0][0])
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)