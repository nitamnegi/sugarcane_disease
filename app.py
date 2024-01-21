from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import io
import os

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_SIZE = 256

# Load the pre-trained model
model = tf.keras.models.load_model('sgr.h5')
class_names = ["Bacterial Blight","Healthy", "Red Rot"] # Replace with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        # Ensure the file has an allowed extension (optional)
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if os.path.exists("static/uploads/uploaded_image.jpg"):
                os.remove("static/uploads/uploaded_image.jpg")

            file.save(file_path)

            # Convert the saved file to an image array
            image = tf.image.decode_image(tf.io.read_file(file_path), channels=3)
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            image = tf.expand_dims(image, 0)

            predictions = model.predict(image)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)

            # Rename the saved file as 'uploaded_image.jpg'
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            os.rename(file_path, new_file_path)

            return render_template('index.html', prediction=f'Predicted class: {predicted_class}, Confidence: {confidence}%')

    return render_template('index.html', prediction='Invalid file format or no file provided.')

if __name__ == '__main__':
    app.run(debug=True)
