from abc import ABC, abstractmethod
from rs_helper.core.Prediction import Prediction


class Model(ABC):
    """
    This abstract method is the parent class of all machine learning models used for predicting the class of
    a supplied problem description
    """
    def __init__(self, path_to_model: str):
        """
        This abstract method is the parent class of all machine learning models used for predicting the class of
        a supplied problem description

        :param path_to_model: Path to the model
        :type path_to_model: str
        """
        if not isinstance(path_to_model, str):
            raise ValueError("The parameter path_to_model must be of type string")

        self.path = path_to_model
        self.model = None

    @abstractmethod
    def initialize(self) -> None:
        """
        The method initialize the model and save it to the class variable model

        :return: None
        """
        pass

    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """
        Each model needs to be able to predict something

        :return: the prediction
        :rtype: Prediction
        """
        pass

    @abstractmethod
    def normalize_result(self, prediction: Prediction) -> Prediction:
        """
        To scale the confidence of the prediction between 0-1. This should guarantee that predictions from different
        models are comparable to each other.

        :param prediction: A prediction made by a model in form of an prediction object
        :type prediction: Prediction

        :return: A prediction object with normalized results
        :rtype: Prediction
        """
        pass
