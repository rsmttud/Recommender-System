from abc import ABC, abstractmethod
from typing import List

from rs_helper.core import Topic
from rs_helper.core import Keyword


class KeywordExtractor(ABC):
    """
    An abstract class which serves as template to implement a new keyword_extraction algorithm.
    """

    def __init__(self):
        self.class_name = self.__class__.__name__

    @abstractmethod
    def extract_keywords(self, *kwargs) -> List[str]:
        """
        Method to create find all keywords and their ranks

        :return: A list of keywords
        :rtype: list(str)
        """
        pass

    @abstractmethod
    def generate_topic(self, token_rank_dict, label: str) -> Topic:
        """
        Method to transfer the keywords found into a Topic object.

        :return: The topic object
        :rtype: Topic
        """
        pass
