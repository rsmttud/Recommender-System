import os
import time
from flask import Flask, render_template, request, json
from helper_functions import save_txt_from_interface
from rs_helper.core import *
from rs_helper.helper import *
from typing import Dict

app = Flask(__name__)


# TODO definetly needs to be adjusted
@app.route('/', methods=["GET"])
def main():
    return render_template("index.html")


@app.route('/classify', methods=["POST"])
def classify():
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


@app.route('/save_json', methods=["POST"])
def save_json():
    save_dir = "data/output"
    file_name = str(time.time()) + ".json"
    # Write Json
    with open(os.path.join(save_dir, file_name), "w") as file:
        json.dump(dict(request.form), file, indent=4)
        file.close()

    response = app.response_class(
        response="True",
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

    path_long_desc = os.path.join("data/input/long_desc", file_name + ".txt")
    facade = RecommendationFacade(path_to_files=path_long_desc)

    if pipeline_method == "classification":
        result = facade.run(classification=True)
        label_map = LabelMap(path_to_json="models/label_maps/4_classes.json")
        for index, class_id in enumerate(result.classes):
            # The float type casting is necessary, because json.dumps doesnt support np.float32
            prediction.update({label_map.get_name(class_id): float(result.values[index])})

    elif pipeline_method == "lda":
        result = facade.run(lda=True)
        label_map = LabelMap(path_to_json="models/label_maps/lda_3_topics.json")
        for index, class_id in enumerate(result.classes):
            # The float type casting is necessary, because json.dumps doesnt support np.float32
            prediction.update({label_map.get_name(class_id): float(result.values[index])})
    elif pipeline_method == "svc_classification":
        result = facade.run(svc_classification=True)
        label_map = LabelMap(path_to_json="models/label_maps/3_classes.json")
        for index, class_id in enumerate(result.classes):
            # The float type casting is necessary, because json.dumps doesnt support np.float32
            prediction.update({label_map.get_name(class_id): float(result.values[index])})
    elif pipeline_method == "key_ex":
        result = facade.run(key_ex=True)
        for index, class_id in enumerate(result.classes):
            prediction.update({class_id: float(result.values[index])})
    # TODO: Clarify how to work with sentence predictions. Multiple predictions need to be placed then!
    elif pipeline_method == "one_to_one_gru_classification":
        result = facade.run(gru_oto=True)
        # Prediction per sentence
        for pred in result:
            for index, class_id in enumerate(pred.classes):
                prediction.update({class_id: float(pred.values[index])})
    else:
        prediction.update({"NoMethod": "Implemented"})

    return prediction


if __name__ == '__main__':
    app.run(debug=True)
