import asyncio
from os import abort

from PIL import Image
from flask import Flask, request, jsonify, make_response

from model import train_model, process_image, MUSHROOMS_NN
import uuid

app = Flask(name)

BASE_PATH = "/mico/api/v1/"

POISONOUS = "poisonous"
POISONOUS_FOLDER = "/poisonous mushroom sporocarp/"

EDIBLE = "edible"
EDIBLE_FOLDER = "/edible mushroom sporocarp/"


@app.route(BASE_PATH + "/classify", methods=["POST"])
def classify_image():
    if not request.json or not 'image' in request.json:
        abort(400)
    file = request.json['image']
    img = Image.open(file.stream)
    mushroom_klass = process_image(img)
    return jsonify({"class": mushroom_klass})

    return jsonify({"class": mushroom_klass}), 201


@app.route(BASE_PATH + "/updateTrainingSet", methods=["POST"])
def add_image():
    if not request.json or not 'image' in request.json or not 'class' in request.json:
        abort(400)
    file = request.json['image']
    klass = request.json['class']
    img = Image.open(file.stream)

    if klass == POISONOUS:
        img.save(MUSHROOMS_NN + POISONOUS_FOLDER + uuid.uuid4() + ".jpg", 'JPEG')
    elif klass == EDIBLE:
        img.save(MUSHROOMS_NN + EDIBLE_FOLDER + uuid.uuid4() + ".jpg", 'JPEG')
    else:
        abort(400)

    return jsonify({"status": "success"})


@app.route(BASE_PATH + "/train", methods=["GET"])
def train():
    train_model()
    return jsonify({"status": "success"})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if name == "main":
    app.run(debug=False, use_reloader=False)