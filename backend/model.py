import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, RobustScaler


class Model:
    def __init__(self, path):
        with open(path, 'rb') as file:
            self._model = joblib.load(file)

    def run(self, input):
        normalised_input = self._normalise(input)
        result = self._model.predict(normalised_input)
        return result

    def _normalise(self, input):
        pp_pipe = Pipeline([
            ("norm", Normalizer()),
            ("scale", RobustScaler())
        ])
        normalised = pp_pipe.fit_transform(input)
        return normalised
