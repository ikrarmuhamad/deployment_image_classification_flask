from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'EfficientNetV2B0.h5'

# Load your trained model
model = load_model(MODEL_PATH)


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    # image_path = "./images/" + imagefile.filename
    # imagefile.save(image_path)
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(imagefile.filename))
    imagefile.save(file_path)
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, size = [224, 224])
    target_class = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger',
                    'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
    pred_prob = model.predict(tf.expand_dims(img, axis = 0))
    pred_class = str(target_class[pred_prob.argmax()])
    classification = f'{pred_class}, our confidence level : {pred_prob.max() * 100:.2f}%'
    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)