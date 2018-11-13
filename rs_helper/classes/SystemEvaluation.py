from rs_helper.classes import Prediction
from sklearn.metrics import confusion_matrix
from typing import *


class SystemEvaluation:
    """
    This class is to evaluate the overall system and it's predicting power
    """

    def __init__(self, predictions: list, labels: list):
        if not isinstance(predictions[0], Prediction):
            raise ValueError("The parameter predictions need to be of type Prediction")
        self.predictions = predictions
        self.labels = labels

        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

        self.positives = self.tn + self.tp
        self.negatives = self.fp + self.fn

        self.__init_metrics()

    # TODO I would use sklearn confusion matrix and use.ravel to get all tn,tp,.. Easier for us, more Eval.
    def calculate_positives_and_negatives(self) -> Tuple[int, int]:
        """
        This method counts all positive and negative predictions
        :return: tuple(int, int)
        """
        pos, neg = 0, 0
        for p, l in zip(self.predictions, self.labels):
            pos += 1 if p.get_max_class() == l else 0
            neg += 1 if p.get_max_class() != l else 0
        self.positives = pos
        self.negatives = neg
        return pos, neg

    def __init_metrics(self):
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true=self.labels, y_pred=self.predictions).ravel()

    def calculate_recall(self):
        return self.tp / (self.tp + self.fn)

    def calculate_true_negative_rate(self):
        return self.tn / (self.tn + self.fp)

    def calculate_precision(self):
        return (self.tp / (self.tp + self.fp))

    def calculate_accuracy(self):
        return (self.tp + self.tn) / (self.positives + self.negatives)

    def get_confusion_matrix(self, labels: List[str] = None):
        if labels:
            return confusion_matrix(y_true=self.labels, y_pred=self.predictions, labels=labels)
        return confusion_matrix(y_true=self.labels, y_pred=self.predictions)

    def save_evaluation(self) -> bool:
        """
        Method to save all evaluation results for future comparison
        :return: boolean
        """
        pass
