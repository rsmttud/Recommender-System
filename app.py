from flask import Flask, render_template, request
from helper_functions import save_txt
from rs_helper.classes import RecommendationFacade

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        problem_title = "" if "title" not in request.form else request.form['title']
        problem_short_desc = "" if "short_description" not in request.form else request.form['short_description']
        problem_long_desc = "" if "long_description" not in request.form else request.form['long_description']

        dir_name = save_txt(problem_title, problem_short_desc, problem_long_desc)

        facade = RecommendationFacade(path_to_files=dir_name)
        scores = facade.run(lda=True)

        return render_template("index.html", params={
            "result": scores
        })

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
