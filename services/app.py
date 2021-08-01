import flask
import random
from flask import Flask, render_template
from PIL import Image
from random import seed
from NoiseService import add_noise
from dto import Response, ResponseImages, ResponsePredications, encodeImage

seed(1)

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

    target = flask.request.form.get('target', '')
    if target == "random":
        target = random.randint(0, 999)
    else:
        target = int(target)

    label_file = "imagenet.txt"  # TODO update it to be the real path
    result = add_noise(pil_img, model, label_file, target)

    result[4][0] = round(result[4][0], 3)
    result[6][0] = round(result[6][0], 3)

    noised_image = result[0]
    original_image = result[1]
    pure_noise = result[2]
    class_no_noise = result[3]
    confidence_no_noise = result[4]
    class_noised = result[5]
    confidence_noised = result[6]

    response = flask.jsonify(
        {
            'predictions': {
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
    )
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.status_code = 200
    return response


if __name__ == '__main__':
    app.run()
