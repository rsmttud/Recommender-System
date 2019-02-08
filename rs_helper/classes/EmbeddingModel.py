from abc import ABC, abstractmethod
from typing import *
import tensorflow as tf
import numpy as np
import pickle
from joblib import load



class EmbeddingModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def initialize_model(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def inference(self, words: List[str]) -> np.ndarray:
        pass


class FastText(EmbeddingModel):
    """
    Loads the FastText model and get the Vectors.
    """

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.model = None
        #self.path = "/Users/Daniel/PycharmProjects/Recommender-System/models/embeddings/FastText/FastText_model_20k_full.joblib"
        self.path = "/Users/Daniel/PycharmProjects/Recommender-System/notebooks/FastText/ft_model_15000.pkl"
        self.initialize_model()

    def initialize_model(self, **kwargs):
        try:
            tf.logging.info("FastText Model is loading")
            if self.path.endswith(".pkl"):
                self.model = pickle.load(open(self.path, "rb"))
            else:
                self.model = load(self.path)
            tf.logging.info("FastText Model loaded!")
        except Exception as e:
            tf.logging.warning("Something went wrong while loading the FastText Model..")
            tf.logging.warning(e)

    def inference(self, words: List[str]) -> np.ndarray:
        embeddings = []
        for word in words:
            if self.model.wv.__contains__(word):
                embeddings.append(self.model.wv.__getitem__(word))
        return np.array(embeddings)
