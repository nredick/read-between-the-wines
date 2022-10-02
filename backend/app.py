# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.10.7 (''venv_maishacks2022'': venv)'
#     language: python
#     name: python3
# ---

# %%
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import model
import encoder
import year_detector


# %%
app = Flask(__name__, template_folder='templates')
_model = model.Model("../models/wine_model.pkl")
_encoder = encoder.Encoder()
detector = year_detector.YearDetector()


# %%
@app.route('/', methods=["GET"])
def main():
    return render_template('main.html')


# %%
@app.route('/wine', methods=["GET"])
def wine():
    year, location = request.args.get('year'), request.args.get('location')
    if year is None or location is None:
        return 'Missing parameters', 400
    try:
        data_frame = _encoder.encode_features(year, location)
    except:
        return "Geoprocessing failed, ensure your location exists and isn't in the water!", 400
    result = _model.run(data_frame)

    print(result, type(result))
    # response = {'text': result}
    # return jsonify(response)

    q = f"The predicted quality of the wine is {result[0][0]:.2f}/100"
    p = f"The predicted price of the wine is ${result[0][1]:.2f}"
    
    return render_template('main.html', quality=q, price=p)
    


# %%
@app.route('/wine_year_by_image', methods=["GET"])
def wine_year_by_image():
    input_json = request.get_json(force=True)
    image = input_json['image']
    year = detector.detect(image)
    return year

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/team', methods=['GET'])
def team():
    return render_template('team.html')


# %%
def run():
    app.run(host='0.0.0.0', port=8888)


# %%
run()
