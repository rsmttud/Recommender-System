from flask import Flask, render_template, request
from helper_functions import save_txt_from_interface
from rs_helper import RecommendationFacade
import os

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        problem_title = "" if "title" not in request.form else request.form['title']
        problem_short_desc = "" if "short_description" not in request.form else request.form['short_description']
        problem_long_desc = "" if "long_description" not in request.form else request.form['long_description']
        problem_method = "" if "method" not in request.form else request.form['method']

        file_name = save_txt_from_interface(problem_title, problem_short_desc, problem_long_desc)
        prediction = []

        # TODO adjust for LDA
        if problem_method == "classification":
            path = os.path.join("data/input/long_desc", file_name + ".txt")
            facade = RecommendationFacade(path_to_files=path)
            result = facade.run(classification=True)
            prediction = zip(result.classes, result.values)

        return render_template("index.html", prediction=prediction, params={
            "title": problem_title,
            "short": problem_short_desc,
            "long": problem_long_desc, })

    else:
        return render_template("index.html")


@app.route("/process_main")
def process_main():
    return render_template("process_main.html")


if __name__ == '__main__':
    app.run(debug=True)
