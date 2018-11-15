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
    def initialize(self) -> None:
        """
        The method initialize the model and save it to the class variable model
        :return:
        """
        pass

    @abstractmethod
    def predict(self) -> Prediction:
        """
        Each model needs to be able to predict something
        :return: prediction object
        """
        pass

    # TODO Documentation necessary
    @abstractmethod
    def normalize_result(self, prediction: Prediction) -> Prediction:
        """
        To
        :param prediction: a prediction made by a model in form of an prediction object
        :return: A prediction object with normalized results
        """
        pass

