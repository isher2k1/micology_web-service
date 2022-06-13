import tensorflow as tf
import keras
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

#reteach()

def answer():
    model = tf.keras.models.load_model('mashrooms_model.h5')

    path = 'mushrooms/test/cv (1).jpeg'
    img = image.load_img(path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    print(float(classes[0]))
    if classes[0]<0.5:
        print("Non-poisoned mashroom")
    else:
        print("Poisoned mashroom")