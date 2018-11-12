

class Prediction:

    def __init__(self, classes: list, values: list):
        if not isinstance(classes, list) or not isinstance(values, list):
            raise ValueError("Parameters classes and values need to be of type list")
        if len(classes) != len(values):
            raise ValueError("Lists classes and values need to have same length")

        self.classes = classes
        self.values = values

    def get_max_class(self):
        max_num = max(self.values)
        return self.classes[self.values.index(max_num)]
