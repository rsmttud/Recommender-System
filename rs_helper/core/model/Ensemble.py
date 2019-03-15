from rs_helper.core.model import Model
from rs_helper.core import Prediction
from typing import *


class Ensemble(Model):

    def __init__(self, weightening_scheme: List[float], n_classes: int = 3) -> None:
        """
        Class to do weighted averaging based on different classifier predictions.
        Given three classes, every weight in list will be used for three elements of the prediction list.

        :param weightening_scheme: The weightening schema. Every classifier needs to have a weight.
        :type weightening_scheme: list(float)
        """
        super().__init__(path_to_model="")
        self.n_classes = n_classes
        self.weights = weightening_scheme
        self.label_maps = None
        self.initialize()

    def initialize(self):
        pass

    def predict(self, predictions: List[Prediction]) -> Prediction:
        """
        Perform the merge of predictions according to the weighted averaging approach.

        :param predictions: List of all predictions of the classifiers to merge
        :type predictions: list(Prediction)

        :return: The merged prediction
        :rtype: Prediction
        """
        if type(predictions[0]) != Prediction:
            raise ValueError("x needs to be of type List(Prediction)")

        compressed = [sorted(x.compress(), key=lambda x: x[0]) for x in predictions]
        x = [[y for x, y in p] for p in compressed]

        if len(x) != len(self.weights):
            raise ValueError("Number of weights does not match number of classifications. "
                             "Please ensure every classifier has its weight.")

        final_predictions = list()
        for c_pred in zip(*x):
            final_predictions.append(sum([c_pred[k] * self.weights[k] for k in range(self.n_classes)]))

        p = Prediction(values=final_predictions, classes=[c for c, p in compressed[0]])
        return p

    def normalize_result(self, prediction: Prediction) -> None:
        pass

