import pickle as pkl
import operator
from functools import reduce

from rs_helper.core.model import Model
from rs_helper.core.Prediction import Prediction
from rs_helper.core.Preprocessor import Preprocessor


class LatentDirichletAllocation(Model):

    def __init__(self, path_to_model: str, path_to_vectorizer: str):
        super().__init__(path_to_model=path_to_model)
        self.vec_path = path_to_vectorizer
        self.vectorizer = None

    def initialize(self):
        self.model = pkl.load(open(self.path, "rb"))
        self.vectorizer = pkl.load(open(self.vec_path, "rb"))

    def predict(self, text: str) -> Prediction:
        raw_data = self.__prepare_data(text)
        vectorized = self.vectorizer.transform(raw_data)
        scores = self.model.transform(vectorized)[0]
        pred = Prediction(classes=[i for i, _ in enumerate(scores)], values=list(scores))
        return pred

    # TODO fix arguments in p.lemmatize
    def __prepare_data(self, text: str) -> list:
        p = Preprocessor(text)
        lemmatized = p.lemmatize(remove_punct_and_nums=True, remove_stops=True)
        return reduce(operator.concat, lemmatized)

    def normalize_result(self, prediction: Prediction):
        pass
