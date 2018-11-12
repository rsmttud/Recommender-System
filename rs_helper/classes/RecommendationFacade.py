from rs_helper.classes import Corpora
from rs_helper.classes import Vectorizer
from rs_helper.classes import Prediction


class RecommendationFacade:

    def __init__(self, path_to_file: str):
        corpora = Corpora(path=path_to_file)

    def run(self, lda: bool = False, key_ex: bool = False, doc2vec: bool = False, classification: bool = False):
        pass

    def __lda_pipeline(self):
        pass

    def __merge_predictions(self, predictions: list) -> Prediction:
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        return Prediction()
