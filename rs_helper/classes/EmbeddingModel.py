from abc import ABC, abstractmethod
from typing import *
import tensorflow as tf
import numpy as np
import pickle
from joblib import load, dump
import json
import os
from gensim.models import FastText


class EmbeddingModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def initialize_model(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def inference(self, words: List[str]) -> np.ndarray:
        pass


class FastTextWrapper(EmbeddingModel):
    """
    Loads the FastText model and get the Vectors.
    """

    def __init__(self, path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.path = path
        if path != "":
            self.initialize_model()
        self.config = dict()
        self.dimensions = 100

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

    def inference(self, words: List[str], sentence_level: bool = False) -> np.ndarray:
        if not self.model:
            raise ValueError("Model needs to be initialized first. Please call FastText() with correct path")
        embeddings = []
        for word in words:
            if self.model.wv.__contains__(word):
                embeddings.append(self.model.wv.__getitem__(word))
        if not sentence_level:
            return np.array(embeddings)
        else:
            return sum(embeddings)/len(embeddings)

    def train(self,
              data: List[List[str]],
              save_path: str,
              save_name: str,
              iterations: int = 5,
              window_size: int = 5,
              min_count: int = 3,
              hs: int = 0):

        if os.path.exists(save_path):
            raise IOError("Save path already exists. Please specify another one to not override.")
        else:
            os.mkdir(save_path)

        self.config.update({"path": save_path, "name": save_name,
                            "epochs": iterations, "window size": window_size,
                            "min count": min_count, "hierarchical softmax": hs})

        model = FastText(data,
                         size=self.dimensions,
                         window=window_size,
                         min_count=min_count,
                         iter=iterations,
                         hs=hs,
                         workers=3,
                         sg=1)

        try:
            dump(model, os.path.join(save_path, save_name))
        except:
            tf.logging.warning("Model could not be saved!")
        return None

    def save_config_json(self, config_path, **kwargs):
        for key, value in kwargs.items():
            self.config.update({str(key): str(value)})
        with open(config_path, "w+") as _json:
            json.dump(self.config, _json)

