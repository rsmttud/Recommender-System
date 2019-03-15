import numpy as np
import datetime
import os
from typing import List
from rs_helper.core import Prediction
from sklearn.metrics import confusion_matrix, recall_score, classification_report, accuracy_score


class SystemEvaluation:
    """
    The purpose of the class is to evaluate the overall system and it's predicting power
    """

    def __init__(self, predictions: List[Prediction], labels: list):
        """

        :param predictions: A List of prediction objects
        :type predictions: list(Prediction)
        :param labels: A list containing the ground-truth labels
        :type labels: list
        """
        if not isinstance(predictions[0], Prediction):
            raise ValueError("The parameter predictions need to be of type Prediction")
        self.predictions = predictions
        self.labels = labels
        self.y_pred = None

        self.confusion_matrix = None

        self.positives = 0
        self.negatives = 0

        self.__init_metrics()
        self.__calculate_positives_and_negatives()

    def __calculate_positives_and_negatives(self) -> None:
        """
        Method return calculates all positives and negative classifications

        :return:
        """
        pos, neg = 0, 0
        for p, l in zip(self.predictions, self.labels):
            pos += 1 if p.get_class_with_max_confidence() == l else 0
            neg += 1 if p.get_class_with_max_confidence() != l else 0
        self.positives = pos
        self.negatives = neg

    def __init_metrics(self) -> None:
        """
        Initializes some attributes of the class

        :return:
        """
        self.y_pred = [x.get_class_with_max_confidence() for x in self.predictions]
        self.confusion_matrix = confusion_matrix(y_true=self.labels, y_pred=self.y_pred)

        # self.tn, self.fp, self.fn, self.tp = self.confusion_matrix.ravel()

    def calculate_accuracy_score(self) -> float:
        """
        Calculates the accuracy according to sklearn.metrics.accuracy_score

        :return: normalized accuracy score
        :rtype: float
        """
        return accuracy_score(self.labels, self.y_pred)

    def calculate_recall(self) -> float:
        """
        Calculates the recall given by the class attribute self.predictions and self.labels.
        Uses the sklean.metrics.recall_score method

        :return: recall as float
        :rtype: float
        """
        return recall_score(self.labels, self.y_pred, average="micro")

    def get_confusion_matrix(self, labels: List[str] = None) -> np.ndarray:
        """
        Returns an sklearn.metrics confusion matrix

        :param labels: the ground truth labels
        :type labels: list(str)

        :return: Numpy array representing the confusion matrix
        :rtype: np.ndarray
        """
        if labels:
            return confusion_matrix(y_true=self.labels, y_pred=self.predictions, labels=labels)
        return confusion_matrix(y_true=self.labels, y_pred=self.predictions)

    def save_evaluation(self) -> bool:
        """
        Method to save all evaluation results for future comparison. Results are saved to the eval folder

        :return: Status
        :rtype: bool
        """
        try:
            date = str(datetime.datetime.today().strftime('%d-%m-%y_%H:%M:%S'))
            dir = "eval"
            str_1 = "# Evaluation of {} \n".format(date)
            str_2 = "## Classification report: \n"
            class_report = classification_report(y_true=self.labels, y_pred=self.y_pred)

            file_name = os.path.join(dir, date + "_" + "eval.md")

            file = open(file_name, "w")
            file.write(str_1 + str_2 + class_report)
            file.close()

            return True
        except:
            return False
