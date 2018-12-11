from flask import Flask, render_template, request, json
from helper_functions import save_txt_from_interface
from rs_helper import RecommendationFacade
from rs_helper import LabelMap
import os
from typing import Dict

app = Flask(__name__)


@app.route('/', methods=["GET"])
def main():
    return render_template("index.html")


@app.route('/classify', methods=["POST"])
def classify():
    print("YUHUUU a POST")
    problem_title = "" if "title" not in request.form else request.form['title']
    problem_short_desc = "" if "short_description" not in request.form else request.form['short_description']
    problem_long_desc = "" if "long_description" not in request.form else request.form['long_description']
    pipeline_method = "" if "method" not in request.form else request.form['method']

    file_name = save_txt_from_interface(problem_title, problem_short_desc, problem_long_desc)
    prediction = get_prediction(pipeline_method, file_name)

    response = app.response_class(
        response=json.dumps(prediction),
        status=200,
        mimetype='application/json'
    )
    return response


def get_prediction(pipeline_method: str, file_name: str) -> Dict:
    """
    Returns a dict {class: value, class: value, ...}, which can be send as json to client.
    :param pipeline_method: The value of the select form
    :param file_name: The file name to the long or short description to construct the corpora
    :return: dict {class: value, class: value, ...}
    """
    prediction = {}

    # TODO adjust for LDA
    if pipeline_method == "classification":
        path_long_desc = os.path.join("data/input/long_desc", file_name + ".txt")
        facade = RecommendationFacade(path_to_files=path_long_desc)
        result = facade.run(classification=True)
        label_map = LabelMap(path_to_json="models/label_maps/4_classes.json")
        for index, class_id in enumerate(result.classes):
            # The float type casting is necessary, because json.dumps doesnt support np.float32
            prediction.update({label_map.get_name(class_id): float(result.values[index])})
    else:
        prediction.update({"NoMethod": "Implemented"})

    return prediction


if __name__ == '__main__':
    app.run(debug=True)
