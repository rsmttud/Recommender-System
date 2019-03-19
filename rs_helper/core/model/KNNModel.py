from rs_helper.core import Prediction
from rs_helper.core.model import Model
from rs_helper.core.distributed_models import EmbeddingModel, DAN, FastTextWrapper
from joblib import load
from typing import Dict, List
from nltk.tokenize import word_tokenize
from rs_helper.core import LabelMap


class KNNModel(Model):
    """
    Class for nearest neighbor classification in vector space against clean dataset data
    """

    def __init__(self, path_to_model: str, embedding_model: EmbeddingModel) -> None:
        """
        Constructor of KNN classifier.

        :param path_to_model: Path to the classifier
        :type path_to_model: str
        :param embedding_model: The embedding model to generate the vector representations
        :type embedding_model: EmbeddingModel
        """
        if not isinstance(embedding_model, DAN) or not isinstance(embedding_model, FastTextWrapper):
            raise ValueError("Embedding model must be of type DAN or FastTextWrapper")

        super().__init__(path_to_model)
        self.embedding_model = embedding_model
        self.initialize()

    def initialize(self) -> None:
        """
        Initializes the model by loading the model from the supplied path

        :return: None
        """
        self.model = load(self.path)

    def predict(self, text: str) -> Prediction:
        """
        :param text: Text to predict.
        :type text: str

        :return: The final prediction
        :rtype: Prediction
        """
        x = self.embedding_model.inference(word_tokenize(text)) if isinstance(self.embedding_model, DAN) \
            else self.embedding_model.inference(word_tokenize(text), sentence_level=True)
        prediction = self.model.predict_proba(x)
        lm = LabelMap("models/label_maps/3_classes.json")

        classes = list()
        values = list()
        for i, y in prediction[0]:
            classes.append(lm.get_name(i))
            values.append(y)

        return Prediction(classes=classes, values=values)

    def normalize_result(self, prediction: Prediction) -> Prediction:
        pass
