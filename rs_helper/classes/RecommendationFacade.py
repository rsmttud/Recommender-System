from rs_helper.classes import Corpora
from rs_helper.classes import Vectorizer
from rs_helper.classes import Prediction

# TODO implement the facade
class RecommendationFacade:
    """
    A facade which contains of single functions for each classification pipeline.
    """

    def __init__(self, path_to_file: str):
        self.corpora = Corpora(path=path_to_file)

    def run(self, lda: bool = False, key_ex: bool = False, doc2vec: bool = False, classification: bool = False):
        pass

    def __lda_pipeline(self):
        pass

    # TODO __merge_predictions() need to be implemented
    """
    def __merge_predictions(self, predictions: list) -> Prediction:
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        pass
        # return Prediction()
    """
