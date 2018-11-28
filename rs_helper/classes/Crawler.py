from abc import ABC
import pickle
import os


class Crawler(ABC):

    def __init__(self, out_path: str):
        self.out_path = out_path
        self.class_name = self.__class__.__name__
        pass

    def crawl(self):
        pass

    def save_to_file(self):
        pass

    def save_to_dataframe(self):
        pass

    def pickle_dataframe(self, df, dir):
        return pickle.dump(df, open("../../../data/crawl_dataframes/" + self.class_name + "_" + dir + "_crawl.pkl", "wb"))
