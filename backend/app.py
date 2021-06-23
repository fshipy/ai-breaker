import flask
from flask import Flask, send_file
from PIL import Image
import io

from backend.NoiseService import add_noise

app = Flask(__name__)


@app.route('/noise/<model>', methods=['POST'])
def noise_endpoint(model):
    imageFile = flask.request.files.get('image', '')
    pil_img = Image.open(imageFile)
    # TODO we may need to change result, it is now a tuple
    target = 2 # TODO update it to user input?
    label_file = "imagenet.txt" # TODO update it to be the real path
    result = add_noise(pil_img, model, label_file, target)
    noised_image = result[0]
    img_io = io.BytesIO()
    noised_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run()
