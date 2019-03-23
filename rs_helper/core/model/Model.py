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
