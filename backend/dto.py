import io
from base64 import encodebytes


def encodeImage(noised_image):
    img_io = io.BytesIO()
    noised_image.save(img_io, 'PNG')
    # encode as base64
    return encodebytes(img_io.getvalue()).decode('ascii')


class ResponseImage:
    def __init__(
            self,
            image
    ):
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        self.image = encodebytes(img_io.getvalue()).decode('ascii')  # encode as base64


class ResponseImages:
    def __init__(
            self,
            noised_image,
            original_image,
            pure_noise
    ):
        self.noised = ResponseImage(noised_image),
        self.original = ResponseImage(original_image),
        self.pure_noise = ResponseImage(pure_noise)


class ResponsePredications:
    def __init__(
            self,
            prediction,
            confidence
    ):
        self.prediction = prediction
        self.confidence = confidence


class Response:
    def __init__(
            self,
            noised_image,
            original_image,
            pure_noise,
            class_no_noise,
            confidence_no_noise,
            class_noised,
            confidence_noised,
    ):
        self.images = ResponseImages(noised_image, original_image, pure_noise)
        self.predictions = [
            {
                'original': ResponsePredications(class_no_noise, confidence_no_noise)
            },
            {
                'noised': ResponsePredications(class_noised, confidence_noised)
            }
        ]
