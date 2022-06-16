import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image


MUSHROOMS_NN = 'mushrooms/nn'
MUSHROOMS_TEST = 'mushrooms/test'
MUSHROOMS_MODEL = 'mushrooms_model.h5'


def train_model():
    mushrooms_data = ImageDataGenerator(rescale=1. / 255)
    test_data = ImageDataGenerator(rescale=1. / 255)

    mushrooms_data = mushrooms_data.flow_from_directory(
        MUSHROOMS_NN,
        target_size=(128, 128),
        batch_size=40,
        class_mode='binary')

    test_data = test_data.flow_from_directory(
        MUSHROOMS_TEST,
        target_size=(128, 128),
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

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

    model.fit(
        mushrooms_data, epochs=1, validation_data=test_data
    )

    model.save(MUSHROOMS_MODEL)


def process_image(img):
    model = tf.keras.models.load_model(MUSHROOMS_MODEL)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    print(float(classes[0]))
    if classes[0] < 0.5:
        return "Non-poisoned mashroom"
    else:
        return "Poisoned mashroom"


if __name__ == '__main__':
    #train_model()
    path = "/Users/ivan/Desktop/flask_app/mushrooms/test/ce (1).jpeg"
    img = image.load_img(path, target_size=(128, 128))
    process_image(img)