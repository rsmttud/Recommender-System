import pandas as pd
import os


def read_data(_dir: str) -> pd.DataFrame:
    """
    Function which reads a dir with subdirs containing strings into an pd.DataFrame object

    :param _dir: Path to dir
    :type _dir: str

    :return: DataFrame with [url, text, class] as columns
    :rtype: pd.DataFrame
    """
    data = {}
    data["url"] = []
    data["text"] = []
    data["class"] = []
    for root, dirs, files in os.walk(_dir):
        for _dir in dirs:
            for txt_file in [x for x in os.listdir(os.path.join(root, _dir)) if x.endswith((".txt", ".TXT"))]:
                # Class name = dir name
                class_name = _dir
                # Read File
                file_name = os.path.abspath(os.path.join(root, _dir, txt_file))
                file = open(file_name, "r")
                txt = file.read()
                file.close()
                data["url"].append(file_name)
                data["text"].append(txt)
                data["class"].append(class_name)
    df = pd.DataFrame.from_dict(data)
    del data
    return df
