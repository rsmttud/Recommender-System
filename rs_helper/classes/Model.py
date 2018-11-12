from abc import ABC, abstractmethod
from rs_helper.classes import Prediction


class Model(ABC):
    """
    This abstract method is the parent class of all machine learning models used for predicting the class of
    a supplied problem description
    """
    def __init__(self, path_to_model: str):
        if not isinstance(path_to_model, str):
            raise ValueError("The parameter path_to_model must be of type string")

        self.path = path_to_model
        self.model = None

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self) -> Prediction:
        pass

    @abstractmethod
    def normalize_result(self, prediction: Prediction):
        pass