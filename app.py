from flask import Flask, render_template, request
from helper_functions import save_txt

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        problem_title = "" if "title" not in request.form else request.form['title']
        problem_short_desc = "" if "short_description" not in request.form else request.form['short_description']
        problem_long_desc = "" if "long_description" not in request.form else request.form['long_description']

        save_txt(problem_title, problem_short_desc, problem_long_desc)

        return render_template("index.html", params={
            "title": problem_title,
            "short": problem_short_desc,
            "long": problem_long_desc
        })

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
