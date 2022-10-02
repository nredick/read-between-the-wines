import joblib


class Model:
    def __init__(self, path):
        with open(path, 'rb') as file:
            self._model = joblib.load(file)

    def run(self, input):
        normalised_input = self._normalise(input)
        result = self._model.predict(normalised_input)
        return result.tolist()

    def _normalise(self, input):
        pp_pipe = joblib.load('../models/pipe.pkl')
        normalised = pp_pipe.transform(input)
        return normalised
