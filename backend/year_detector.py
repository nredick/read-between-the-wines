import re
import boto3
import base64


class YearDetector:
    def __init__(self):
        self._client = boto3.client('rekognition')

    def detect(self, image):
        text = self._text_in_image(image)
        year = self._year_in_text(text)
        return year

    def _text_in_image(self, image):
        encoded_image = self._encode_image(image)
        text = self._extract_text(encoded_image)
        return text

    def _year_in_text(self, text):
        for value in text:
            matches = re.findall(r'^(?:19|20)\d{2}$', value)
            if len(matches) != 0:
                return matches[0]
        return None

    def _encode_image(self, image):
        encoded = base64.b64encode(image)
        binary = base64.decodebytes(encoded)
        return binary

    def _extract_text(self, encoded_image):
        request = {'Bytes': encoded_image}
        response = self._client.detect_text(Image=request)
        text = [detection['DetectedText'] for detection in response['TextDetections']]
        return text
