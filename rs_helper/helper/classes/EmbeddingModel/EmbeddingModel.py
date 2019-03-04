from abc import ABC, abstractmethod
from typing import *
import tensorflow as tf
import numpy as np
import pickle
from joblib import load, dump
import os
from gensim.models import FastText
import json


class EmbeddingModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def initialize_model(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def inference(self, words: List[str]) -> np.ndarray:
        pass

    def inference_batches(self, words: List[List[str]]) -> List:
        pass

