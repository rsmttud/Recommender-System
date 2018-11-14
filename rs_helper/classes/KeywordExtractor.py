from abc import ABC, abstractmethod
from rs_helper.classes import Corpora
from rs_helper.classes import Topic
from rs_helper.classes import Keyword


class KeywordExtractor(ABC):

    def __init__(self):
        self.class_name = self.__name__

    @abstractmethod
    def extract_keywords(self) -> list:
        """
        Method to create find all keywords and their ranks
        :return: List(Keyword)
        """
        pass

    @abstractmethod
    def generate_topic(self) -> Topic:
        """
        Method to transfer the keywords found into a Topic object.
        :return: Topic
        """
        pass
