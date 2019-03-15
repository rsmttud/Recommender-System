from typing import List, Tuple
from math import log


# TODO needs a review
class Prediction:
    """
    Class to store the final prediction of pipelines
    """

    def __init__(self, classes: List[str], values: List[float]) -> None:
        """
        Class to store the final prediction of pipelines

        :param classes: List of all available classes
        :type classes: list(str)
        :param values: List of all values for all classes
        :type values: list(float)
        """
        if not isinstance(classes, List) or not isinstance(values, List):
            raise ValueError("Parameters classes and values need to be of type list")
        if len(classes) != len(values):
            raise ValueError("Lists classes and values need to have same length")

        self.classes = classes
        self.values = values

    def __repr__(self):
        return "{}: Classes: {}, Values: {}".format(type(self), self.classes, self.values)

    def get_data_frame(self):
        """
        Returns prediction object as DataFrame

        :return: Prediction as DataFrame
        :rtype: pd.DataFrame
        """
        pass

    def get_class_with_max_confidence(self) -> str:
        """
        Returns the prediction with the highest confidence

        :return: the class name
        :rtype: str
        """
        max_num = max(self.values)
        return str(self.classes[self.values.index(max_num)])

    def scale_log(self) -> None:
        """
        scales the probabilities logarithmic

        :return: None
        """
        # Problem are probabilities below 1
        self.values = [log(1.01 + x, 2) for x in self.values]

    def round_values(self) -> None:
        """
        Rounds the values to three decimal nums

        :return: None
        """
        self.values = [round(x, 3) for x in self.values]

    def compress(self) -> List[Tuple[str, float]]:
        """
        Compresses the prediction in tuples. I.e. [('class A', 0.762), ('class B', 0.77)]

        :return: the compressed tuples
        :rtype: list(tupel(str, float))
        """
        return list(zip(self.classes, self.values))
