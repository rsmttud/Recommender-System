

class Prediction:
    """
    Class to store the final prediction of pipelines
    """
    def __init__(self, classes: list, values: list):
        """
        :param classes: List of all available classes (List(string))
        :param values: List of all values for all classes (List(float))
        """
        if not isinstance(classes, list) or not isinstance(values, list):
            raise ValueError("Parameters classes and values need to be of type list")
        if len(classes) != len(values):
            raise ValueError("Lists classes and values need to have same length")

        self.classes = classes
        self.values = values

    def get_class_with_max_confidence(self) -> str:
        """
        Returns the prediction with the highest confidence
        :return: the class name as string
        """
        max_num = max(self.values)
        return str(self.classes[self.values.index(max_num)])
