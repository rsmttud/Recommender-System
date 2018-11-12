from rs_helper.classes import Prediction


class SystemEvaluation:
    """
    This class is to evaluate the overall system and it's predicting power
    """
    def __init__(self, predictions: list, labels: list):
        if not isinstance(predictions[0], Prediction):
            raise ValueError("The parameter predictions need to be of type Prediction")
        self.predictions = predictions
        self.labels = labels
        self.positives = 0
        self.negatives = 0

    def calculate_positives_and_negatives(self):
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

    def calculate_recall(self):
        pass

    def calculate_accurracy(self):
        pass

    def save_evaluation(self) -> bool:
        """
        Method to save all evaluation results for future comparison
        :return: boolean
        """
        pass
