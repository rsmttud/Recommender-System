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
    if n == 1:
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)