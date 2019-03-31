import os
import tensorflow as tf
import numpy as np
import pickle
import json
from joblib import load, dump
from gensim.models import FastText
from typing import *
from rs_helper.core.distributed_models.EmbeddingModel import EmbeddingModel


class FastTextWrapper(EmbeddingModel):
    """
    Class to train and use FastText Models
    """

    def __init__(self, path: str = "", **kwargs) -> None:
        """
        Class to train and use FastText Models

        :param path: Path to the .joblib dumped model
        :type path: str
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.model = None
        self.path = path
        if path != "":
            self.initialize_model()
        self.config = dict()
        self.dimensions = 100

    def initialize_model(self, **kwargs) -> None:
        """
        Initializes the model by loading it via joblib.load()

        :param kwargs:
        :return: None
        """
        try:
            tf.logging.info("FastText model is loading")
            if self.path.endswith(".pkl"):
                self.model = pickle.load(open(self.path, "rb"))
            else:
                self.model = load(self.path)
            tf.logging.info("FastText model loaded!")
        except Exception as e:
            tf.logging.warning("Something went wrong while loading the FastText model..")
            tf.logging.warning(e)

    def inference(self, words: List[str], sentence_level: bool = False) -> np.ndarray:
        """
        Infer a vector for given list of words. Sentence level will merge the word vectors by mean.

        :param words: List of words to infere vectors from
        :type words: list(str)
        :param sentence_level: Weather to receive sentence vector
        :type sentence_level: bool


        :return: The word/sentence vector representations
        :rtype: np.ndarray
        """
        if not self.model:
            raise ValueError("model needs to be initialized first. Please call FastText() with correct path")
        embeddings = []
        for word in words:
            if self.model.wv.__contains__(word):
                embeddings.append(self.model.wv.__getitem__(word))
        if not sentence_level:
            return np.array(embeddings)
        else:
            return sum(embeddings) / len(embeddings)

    def train(self,
              data: List[List[str]],
              save_path: str,
              save_name: str,
              iterations: int = 5,
              window_size: int = 5,
              min_count: int = 3,
              hs: int = 0) -> None:
        """
        Train a new FastText model.

        :param data: The training data
        :type data: list(list(str))
        :param save_path: Directory path to save the model
        :type save_path: str
        :param save_name: Name of the model (Should be <name>.joblib)
        :type save_name: str
        :param iterations: Epochs of the model to train
        :type iterations: int
        :param window_size: Window size used by the model
        :type window_size: int
        :param min_count: Minimal frequency of words to be regarded
        :type min_count: int
        :param hs: Hierarchical Softmax
        :type hs: int

        :return: None
        """
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
            tf.logging.warning("model could not be saved!")
        return None

    def save_config_json(self, config_path: str, **kwargs) -> None:
        """
        Save the config of the model as json

        :param config_path: Path where the config will be stored
        :type config_path: str
        :param kwargs:

        :return: None
        """
        for key, value in kwargs.items():
            self.config.update({str(key): str(value)})
        with open(config_path, "w+") as _json:
            json.dump(self.config, _json)
