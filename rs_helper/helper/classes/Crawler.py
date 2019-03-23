from abc import ABC, abstractmethod
import pickle
from typing import Any
import pandas as pd

class Crawler(ABC):
    """
    Abstract class for all crawlers
    """
    def __init__(self, out_path: str):
        self.out_path = out_path
        self.class_name = self.__class__.__name__
        pass

    @abstractmethod
    def crawl(self) -> None:
        """
        Start crawling ...

        :return: None
        """
        pass

    @abstractmethod
    def save_to_file(self) -> None:
        """
        Save documents to file

        :return: None
        """
        pass

    @abstractmethod
    def save_to_dataframe(self) -> pd.DataFrame:
        """
        Save crawl results to pandas DataFrame

        :return: the Dataframe
        :rtype: pd.DataFrame
        """
        pass

    def pickle_dataframe(self, df: pd.DataFrame, dir: str) -> Any:
        """
        Dump the DataFrame of crawl results with pickle

        :param df: The dataframe to dump
        :type df: pd.DataFrame
        :param dir: Path where the DataFrame should be stored
        :type dir: str

        :return: Status
        :rtype: bool
        """
        return pickle.dump(df, open("../../../data/crawl_dataframes/" + self.class_name + "_" + dir + "_crawl.pkl", "wb"))
