import os
from flask import Flask, render_template, request, json, jsonify
from helper_functions import *
from rs_helper.core import *
from rs_helper.helper import *

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
    # pipeline_method = "" if "method" not in request.form else request.form['method']

    file_name = save_txt_from_interface(problem_title, problem_short_desc, problem_long_desc)
    prediction = get_prediction("", file_name)

    entities = get_entities(problem_short_desc, problem_long_desc)

    response = app.response_class(
        response=json.dumps({"forecast": prediction, "entities": entities}),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/save_json', methods=["POST"])
def save_json():
    save_dir = "data/output"
    file_name = str(int(time.time())) + ".json"
    # Write Json
    with open(os.path.join(save_dir, file_name), "w", encoding="utf-8") as file:
        json.dump(request.json, file, indent=4)

    entities = request.json["entities"]
    class_name = request.json["final_class_agreement"]  # class from forecast with highest prob
    short_desc = request.json["short_desc"]
    long_desc = request.json["long_desc"]

    notebook_path, html_path = create_notebooks(entities, class_name, short_desc, long_desc)
    return jsonify(n_link=notebook_path, h_link=html_path)


if __name__ == '__main__':
    # Use this command for the docker build!
    #app.run(host="0.0.0.0", port=80, debug=False)
    # Use this line for development!
    app.run(debug=True)
