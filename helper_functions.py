import os
import time


def save_txt_from_interface(title:str, short_desc:str, long_desc:str = None):
    """
    Functions saves the corresponding short and long description from the frontend to the data/input folder. The long_desc is optional
    :param title: The title of the problem description
    :param short_desc: A short description in form of an string
    :param long_desc: A long description in form of an string
    :return:
    """
    print(time.time())
    file_name = title + "_" +str(int(time.time()))
    path_short = os.path.join("data/input/short_desc", file_name + ".txt")

    with open(path_short, "w") as file:
        file.write(short_desc)
        file.close()

    if long_desc:
        path_long = os.path.join("data/input/long_desc", file_name + ".txt")
        with open(path_long, "w") as file:
            file.write(long_desc)
            file.close()


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
