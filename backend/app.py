import flask
import model
import year_detector


app = flask.Flask(__name__)
_model = model.Model("/some/path")  # path to the model


@app.route('/')
def base():
    return 'Please specify API'


@app.route('/api/wine', methods=["POST"])
def wine():
    input_json = flask.request.get_json(force=True)
    result = _model.run(input_json)
    response = {'text': result}
    return flask.jsonify(response)


@app.route('/api/wine_year_by_image', methods=["POST"])
def wine_year_by_image():
    detector = year_detector.YearDetector()
    input_json = flask.request.get_json(force=True)
    image = input_json['image']
    year = detector.detect(image)
    return year


def run():
    app.run(port=8888, ssl_context='adhoc')


run()
