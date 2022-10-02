import joblib


class Model:
    def __init__(self, path):
        with open(path, 'rb') as file:
            self._model = joblib.load(file)

    def run(self, input):
        result = self._model.predict(input)
        return result
