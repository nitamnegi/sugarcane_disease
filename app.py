from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_SIZE = 256

model = tf.keras.models.load_model('sgr.h5')
model.compile()
class_names = ["Bacterial Blight","Healthy","Mosaic", "RedRot","Rust", "Yellow"] # Replace with your actual class names

prevention_info = {
    "Bacterial Blight": "Rotate sugarcane with other non-host crops to break the disease cycle. \n Avoid overhead irrigation. \n Regularly monitor sugarcane fields for signs of bacterial blight, such as leaf lesions or wilting.",
    "Healthy": "No prevention measures needed for Healthy plants.",
    "Mosaic": "Start with healthy, virus-free planting material. \n Practice good field sanitation by removing and destroying any infected or symptomatic sugarcane plants. \n Maintain isolation distances between sugarcane fields and other crops or sources of infection.",
    "RedRot": "Do not take ratoon of affected crop . \n Crop rotation in affected fields. \n Water management during rainy season.",
    "Rust": "Planting sugarcane varieties that are resistant or tolerant to rust. \n Proper field sanitation is crucial for rust prevention. \n Maintain balanced nutrient levels in the soil and ensure that sugarcane plants receive proper nutrition.",
    "Yellow": "Conduct regular soil tests to assess nutrient levels, pH, and other soil properties. \n Control pests such as nematodes and insects that can damage sugarcane roots and contribute to yellowing symptoms. \n Monitor sugarcane fields regularly for signs of yellowing, nutrient deficiencies, diseases, and other stress factors."
}

@app.route('/')
def index():
    return render_template('land.html')


@app.route('/index')
def show_index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if os.path.exists("static/uploads/uploaded_image.jpg"):
                os.remove("static/uploads/uploaded_image.jpg")

            file.save(file_path)

            image = tf.image.decode_image(tf.io.read_file(file_path), channels=3)
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.expand_dims(image, 0)

            predictions = model.predict(image)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)

            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            os.rename(file_path, new_file_path)
            
            prevention_text = prevention_info.get(predicted_class, "Prevention information not available.")

            return render_template('index.html', prediction=f'Predicted class: {predicted_class}, Confidence: {confidence}%', prevention=prevention_text)

    return render_template('index.html', prediction='Invalid file format or no file provided.', prevention=None)


if __name__ == '__main__':
    app.run(debug=True)
