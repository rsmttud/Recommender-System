from abc import ABC, abstractmethod
from rs_helper.classes import Corpora
from rs_helper.classes import Topic
from rs_helper.classes import Keyword


class KeywordExtractor(ABC):

    def __init__(self):
        self.class_name = self.__class__.__name__

    def extract_keywords(self, *kwargs) -> list:
        """
        Method to create find all keywords and their ranks
        :return: List(Keyword)
        """
        pass

    def __generate_topic(self, token_rank_dict, label: str) -> Topic:
        """
        Method to transfer the keywords found into a Topic object.
        :return: Topic
        """
        pass
