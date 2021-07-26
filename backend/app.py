
import flask
import random
from flask import Flask
from PIL import Image
from random import seed
from random import randint

from backend.NoiseService import add_noise
from backend.dto import Response, ResponseImages, ResponsePredications, encodeImage

app = Flask(__name__)
seed(1)

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
