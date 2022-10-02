import flask
import model
import encoder
import year_detector


app = flask.Flask(__name__)
_model = model.Model("/models/wine-model.joblib")
_encoder = encoder.Encoder()
detector = year_detector.YearDetector()


@app.route('/')
def base():
    return 'Please specify API'


@app.route('/api/wine', methods=["POST"])
def wine():
    input_json = flask.request.get_json(force=True)
    data_frame = _encoder.encode_features(input_json['year'], input_json['location'])
    result = _model.run(data_frame)
    response = {'text': result}
    return flask.jsonify(response)


@app.route('/api/wine_year_by_image', methods=["POST"])
def wine_year_by_image():
    input_json = flask.request.get_json(force=True)
    image = input_json['image']
    year = detector.detect(image)
    return year


def run():
    app.run(port=8888, ssl_context='adhoc')


run()
