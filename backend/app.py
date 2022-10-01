import flask
import model


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


def run():
    app.run(port=8888, ssl_context='adhoc')


run()
