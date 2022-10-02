import flask
import model
import encoder
import year_detector
import chart


app = flask.Flask(__name__)
_model = model.Model("../models/wine_model.pkl")
_encoder = encoder.Encoder()
detector = year_detector.YearDetector()


@app.route('/')
def base():
    return flask.render_template('main.html')


@app.route('/wine', methods=["GET"])
def wine():
    year, location = flask.request.args.get('year'), flask.request.args.get('location')
    if year is None or location is None:
        return 'Missing parameters', 400
    try:
        data_frame = _encoder.encode_features(year, location)
    except:
        return "Geoprocessing failed, ensure your location exists and isn't in the water!", 400
    result = _model.run(data_frame)
    response = {'text': result}
    return flask.jsonify(response)


@app.route('/wine_year_by_image', methods=["GET"])
def wine_year_by_image():
    input_json = flask.request.get_json(force=True)
    image = input_json['image']
    year = detector.detect(image)
    return year


@app.route('/maps', methods=["POST"])
def maps():
    location = flask.request.args.get('location')
    if location is None:
        return 'Missing parameters', 400
    dataframe = chart.full_data_frame()
    latitude, longitude = _encoder.geoencode(location)
    plot = chart.map_wineries(latitude, longitude, dataframe)
    html = chart.plot_html(plot)
    return html


def run():
    app.run(host='0.0.0.0', port=8888, ssl_context='adhoc')


run()
