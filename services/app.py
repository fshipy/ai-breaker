
import flask
import os
from flask import Flask, render_template
from PIL import Image

from NoiseService import add_noise
from dto import Response, ResponseImages, ResponsePredications, encodeImage

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/impact")
def impact():
    return render_template("impact.html")

@app.route("/try")
def tryme():
    return render_template("try.html")

@app.route('/noise/<model>', methods=['POST'])
def noise_endpoint(model):
    imageFile = flask.request.files.get('image', '')
    pil_img = Image.open(imageFile)

    target = 2  # TODO update it to user input?
    label_file = "imagenet.txt"  # TODO update it to be the real path
    result = add_noise(pil_img, model, label_file, target)

    noised_image = result[0]
    original_image = result[1]
    pure_noise = result[2]
    class_no_noise = result[3]
    confidence_no_noise = result[4]
    class_noised = result[5]
    confidence_noised = result[6]

    return {

        'predictions:': {
            'original': {
                'class': class_no_noise,
                'confidence': confidence_no_noise
            },
            'noised': {
                'class': class_noised,
                'confidence': confidence_noised
            }
        },
        'images': {
            'noised': encodeImage(noised_image),
            'original': encodeImage(original_image),
            'pure_noise': encodeImage(pure_noise)
        }
    }


if __name__ == '__main__':
    app.run()
