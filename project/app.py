from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Muat model CNN
model_path = 'model/my_model.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise ValueError(f'Model file not found at path: {model_path}')

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Ubah sesuai dengan ukuran input model Anda
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalisasi gambar
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def predict(image_path):
    try:
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        # Asumsikan model mengembalikan probabilitas untuk setiap kelas
        class_names = ['Blackspot', 'Cancer', 'Sehat']
        predicted_class = class_names[np.argmax(prediction)]
        return predicted_class
    except Exception as e:
        print(f"Error predicting image: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    image_urls = []
    if request.method == 'POST':
        try:
            image_files = request.files.getlist('images')
            for image_file in image_files:
                if image_file:
                    filename = secure_filename(image_file.filename)
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image_file.save(image_path)
                    prediction = predict(image_path)
                    image_url = url_for('static', filename='uploads/' + filename)
                    predictions.append((image_url, prediction))
        except Exception as e:
            print(f"Error handling upload: {e}")
            return redirect(url_for('index'))
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)



