from rs_helper.classes import Corpora
from rs_helper.classes import Vectorizer
from rs_helper.classes import Prediction


class RecommendationFacade:

    def __init__(self):
        pass

    def __lda_pipeline(self):
        pass

    def __merge_predictions(self, predictions: list) -> Prediction:
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        pass
