

class Model:
    def __init__(self, path):
        self._model = path  # Load model here

    def run(self, input):
        # result = self._model.process(input)
        result = f"No results yet :(\nRequest is: {input}"
        return result
