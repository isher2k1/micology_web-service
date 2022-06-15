import os
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route("/")
def hello_world():
    print(1+2)
    return "<h1>Hello, my cute friend!</h1>"

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def reteach():
    mashrooms_data = ImageDataGenerator(rescale=1./255)
    test_data = ImageDataGenerator(rescale=1./255)

    mashrooms_data = mashrooms_data.flow_from_directory(
        'mushrooms/nn',
        target_size=(128,128),
        batch_size=40,
        class_mode='binary')

    test_data = test_data.flow_from_directory(
        'mushrooms/test',
        target_size=(128,128),
        batch_size=10,
        class_mode='binary')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2), 2),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), 2),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['accuracy'])

    model.fit(
        mashrooms_data,
        epochs=30,
        validation_data = test_data
        )

    model.save('mashrooms_model.h5')
    return("Status: success!")

#reteach()

def answer(fname):
    model = tf.keras.models.load_model('mashrooms_model.h5')

    path = 'mushrooms/upload/' + fname
    img = image.load_img(path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    print(float(classes[0]))
    if classes[0]<0.5:
        return("Non-poisoned mashroom")
    else:
        return("Poisoned mashroom")  
  

from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

class MyForm(FlaskForm):
    file = FileField()


UPLOAD_FOLDER = 'mushrooms/upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return answer(filename)
            
    return '''
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micology</title>
    <link rel="stylesheet" href="../styles/style.css">
    
    </head>
    <body style="background-color:#bcaba0;">
    <div class="container" style="margin: 0 auto; width=500px">
        <header>
            <h1>Micology</h1>
        </header>

        <main>
            <div class="first"><p>Want to know is it dangerous?</p></div> <br>
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
             <br> <br>
            <a href="#">Want to add your photos to our dataset?</a>
            <br><br><br>
            <p>This web service was created to help you<br>distinguish between poisoned and<br>non-poisoned mushrooms.</p>
        </main>

        <footer>
            <p>Thank you for using our service!</p>
        </footer>
    </div>
    </body>
    '''

    @app.route('/reteach', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return reteach()
            
    return '''
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micology</title>
    <link rel="stylesheet" href="../styles/style.css">
    
    </head>
    <body style="background-color:#bcaba0;">
    <div class="container" style="margin: 0 auto; width=500px">
        <header>
            <h1>Micology</h1>
        </header>

        <main>
            <div class="first"><p>Want to increase our dataset?</p></div> <br> 
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
             <br> <br>
            <a href="#>Want to know is it dangerous?</a>
            <br><br><br>
            <p>This web service was created to help you<br>distinguish between poisoned and<br>non-poisoned mushrooms.</p>
        </main>

        <footer>
            <p>Thank you for using our service!</p>
        </footer>
    </div>
    </body>
    '''