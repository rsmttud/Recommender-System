from joblib import load
import operator
import os
from functools import reduce
from rs_helper.core import LabelMap
from rs_helper.core.model import Model
from rs_helper.core.Prediction import Prediction
from rs_helper.core.Preprocessor import Preprocessor


class LatentDirichletAllocation(Model):
    """
    Class for using the trained LDA Model to predict the input text according to its topic distribution
    """
    def __init__(self, path_to_model: str, path_to_vectorizer: str) -> None:
        """
        Constructor of the LDA class

        :param path_to_model: Path to the .joblib model file
        :type path_to_model: str
        :param path_to_vectorizer: Path to the .joblib sklearn vectorizer file
        :type path_to_vectorizer: str
        """
        if not os.path.exists(path_to_vectorizer) or not os.path.exists(path_to_vectorizer):
            raise FileExistsError("Either the model or the vectorizer path seems to be wrong.")
        if not os.path.exists(os.path.join(os.path.dirname(path_to_vectorizer), "label_map.json")):
            raise FileExistsError("In the model directory is no label_map. "
                                  "Please copy the right label_map to the model directory.")
        super().__init__(path_to_model=path_to_model)
        self.vec_path = path_to_vectorizer
        self.label_map = LabelMap(os.path.join(os.path.dirname(path_to_vectorizer), "label_map.json"))
        self.vectorizer = None

    def initialize(self) -> None:
        """
        Initializes the model by loading the model and its vectorizer

        :return: None
        """
        self.model = load(self.path)
        self.vectorizer = load(self.vec_path)

    def predict(self, text: str) -> Prediction:
        raw_data = self.__prepare_data(text)
        vectorized = self.vectorizer.transform(raw_data)
        scores = self.model.transform(vectorized)[0]
        pred = Prediction(classes=[self.label_map.get_name(i) for i, _ in enumerate(scores)], values=list(scores))
        return pred

    def __prepare_data(self, text: str) -> list:
        """
        Preprocess the data by lemmatizing it, removing stopwords, punctuation and numbers

        :param text: The text to process
        :type text: str

        :return: The preprocessed text
        :rtype: list(str)
        """
        p = Preprocessor(text)
        lemmatized = p.transform(remove_nums=True, remove_punct=True)
        return reduce(operator.concat, lemmatized)

    def normalize_result(self, prediction: Prediction):
        pass
