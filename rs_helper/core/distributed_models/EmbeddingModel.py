from abc import ABC, abstractmethod
from typing import *
import numpy as np


class EmbeddingModel(ABC):
    """
    Abstract class for distributed models to create vector representation of text data
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def initialize_model(self, **kwargs) -> Any:
        """
        Initialization of the respective model

        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def inference(self, words: List[str]) -> np.ndarray:
        """
        Inference of embeddings based on words.

        :param words: The words to infer the vectors
        :type words: list(str)

        :return: the embeddings
        :rtype: np.ndarray
        """
        pass

    def inference_batches(self, words: List[List[str]]) -> List:
        pass

