from abc import ABC, abstractmethod
from typing import *
import numpy as np


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

