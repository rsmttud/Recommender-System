import os
import time
import json
from typing import *
from rs_helper import Prediction, RecommendationFacade, EntityExtractor


def save_txt_from_interface(title: str, short_desc: str, long_desc: str = None) -> str:
    """
    Functions saves the corresponding short and long description from the frontend to the data/input folder. The long_desc is optional
    :param title: The title of the problem description
    :param short_desc: A short description in form of an string
    :param long_desc: A long description in form of an string
    :return:
    """
    file_name = title + "_" + str(int(time.time()))
    path_short = os.path.join("data/input/short_desc", file_name + ".txt")

    with open(path_short, "w") as file:
        file.write(short_desc)
        file.close()

    if long_desc:
        path_long = os.path.join("data/input/long_desc", file_name + ".txt")
        with open(path_long, "w") as file:
            file.write(long_desc)
            file.close()

    path_joined = os.path.join("data/input/joined", file_name + ".txt")
    with open(path_joined, "w") as file:
        if long_desc is not None:
            file.write(short_desc + ". " + long_desc)
            file.close()
        else:
            file.write(short_desc)
            file.close()
    return file_name


def find_nth(string, substring, n):
    """
    :param string: String to search in
    :param substring: String to find
    :param n: int (Which occurrence)
    :return: int (Index of the occurrence)
    Function to find the n-th occurrence of a needle in a string. (Used by ScienceDirect Crawler)
    """
    if n == 1:
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)


def generate_dataset_of_files(files: list, class_name: str):
    """
    :param files: List(str) List of path files which should be merged to one single file in data/datasets/ directory
    :param class_name: String (Class name and name of subdir in data/datasets/
    :return: void
    If directory of class does not exist it will be created
    """
    complete = []
    for file in files:
        content = open(file, "r").read()
        if content not in complete:
            complete.append(content)
    if not os.path.exists("data/datasets/{}".format(class_name)):
        os.mkdir("data/{}".format(class_name))
    out_file = open("data/datasets/{}/{}.txt".format(class_name, class_name), "w")
    out_file.write(" ".join(complete))
    out_file.close()


def create_notebooks(entities: List[str], class_name: str, short_desc: str, long_desc: str) -> Tuple[str, str]:
    """
    Methods takes the parameters from the frontend and run the jupyter notebooks regarding to the given parameters.
    :param entities: All extracted entities as List[str]
    :param class_name: Class Name (Recommended Method)
    :param short_desc: str
    :param long_desc: str
    :return: the link to the executed notebook as str
    """

    # Handle less then two entities
    if len(entities) < 1:
        entities = ["entity 1", "entity 2"]
    elif len(entities) == 1:
        entities = [entities[0], "entity 1"]
    # notebook path
    notebook = os.path.join("static/notebooks", class_name + ".ipynb")

    if os.path.isfile(notebook):

        # manipulate notebook
        n_data = mainpulate_notebooks(notebook, class_name, entities, short_desc, long_desc)
        # Save Notebook

        with open(notebook, "w") as json_file:
            json.dump(n_data, json_file)

        # Exec Notebook

        os.system("jupyter nbconvert --execute --to notebook --inplace {}".format(os.path.abspath(notebook)))
        os.system("jupyter nbconvert --to html {}".format(os.path.abspath(notebook)))

        # Doesnt seem to work in docker..
        """
        p = subprocess.Popen("jupyter nbconvert --execute --to notebook --inplace {}".format(notebook))
        p.wait()
        p = subprocess.Popen("jupyter nbconvert --to html {}".format(notebook))
        p.wait()
        """

        # Return File Path
        html = os.path.join("static/notebooks", class_name + ".html")
        return notebook, html
    else:
        return "#!", "#!"


def mainpulate_notebooks(path: str, class_name, entities, short_desc, long_desc) -> dict:
    """
    Helper Function to manipulate your notebooks. Please keep in mind that this is a really specific task - so change
    the method according to your classes and notebooks, if you want.
    :param path: path to notebook
    :return:
    """

    # Loading Notebook
    with open(path) as json_file:
        n_data = json.load(json_file)

    # Class and notebook specific changes
    if class_name in ("clustering", "classification", "pattern_mining", "regression"):
        # Recommended Approach
        n_data["cells"][2]["source"] = "__Recommended Machine Learning Approach__: {} <hr>".format(class_name)
        # Descriptions
        n_data["cells"][3]["source"] = "__Short Description__: {}  <hr>".format(short_desc["text"])
        n_data["cells"][4]["source"] = "__Long Description__: {}  <hr>".format(long_desc["text"])
        # Entities
        _str_entities = ["<li>" + x + "</li>" for x in entities]
        _str_entities.insert(0, "<ul>")
        _str_entities.append("</ul>")
        n_data["cells"][6]["source"] = _str_entities
        # Entity vars
        n_data["cells"][9]["source"] = "var_entity_1 = \"{}\"".format(entities[0])
        n_data["cells"][10]["source"] = "var_entity_2 = \"{}\"".format(entities[1])

        return n_data

    else:
        return n_data


def get_prediction(pipeline_method: str, file_name: str) -> Dict:
    """
    Returns a dict {class: value, class: value, ...}, which can be send as json to client.
    :param pipeline_method: The value of the select form if you want an select box..
    :param file_name: The file name to the long or short description to construct the corpora
    :return: dict {class: value, class: value, ...}
    """
    prediction = {}

    path_long_desc = os.path.join("data/input/joined", file_name + ".txt")
    facade = RecommendationFacade(path_to_files=path_long_desc)

    result = facade.recommend()
    result.scale_log()
    result.round_values()
    for (c, p) in result.compress():
        prediction.update({c: p})

    return prediction


def get_entities(short_desc: str, long_desc: str = "") -> List[str]:
    """
    A method to extract the entities from the given input text.
    :param short_desc: Short description (mandatory) from input
    :param long_desc: Long description from input
    :return: Entities in form of List[str]
    """
    if long_desc == "":
        entity_extractor = EntityExtractor(short_desc)
    else:
        entity_extractor = EntityExtractor(short_desc + ". " +long_desc)
    entities = entity_extractor.extract_entities()
    return entities
