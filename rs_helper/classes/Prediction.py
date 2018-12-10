from typing import *

class Prediction:
    """
    Class to store the final prediction of pipelines
    """
    def __init__(self, classes: List, values: List):
        """
        :param classes: List of all available classes (List(string))
        :param values: List of all values for all classes (List(float))
        """
        if not isinstance(classes, List) or not isinstance(values, List):
            raise ValueError("Parameters classes and values need to be of type list")
        if len(classes) != len(values):
            raise ValueError("Lists classes and values need to have same length")

        self.classes = classes
        self.values = values

    def __repr__(self):
        return "{}: Classes: {}, Values: {}".format(type(self),self.classes, self.values)

    def get_class_with_max_confidence(self) -> str:
        """
        Returns the prediction with the highest confidence
        :return: the class name as string
        """
        max_num = max(self.values)
        return str(self.classes[self.values.index(max_num)])
