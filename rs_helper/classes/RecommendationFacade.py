from rs_helper.classes import Corpora
from rs_helper.classes import Vectorizer
from rs_helper.classes import Prediction
from rs_helper.classes.Preprocessor import Preprocessor
import pickle
import numpy as np


# TODO implement the facade
class RecommendationFacade:
    """
    A facade which contains of single functions for each classification pipeline.
    """

    def __init__(self, path_to_files: str):
        self.corpora = Corpora(path=path_to_files)

    def run(self, lda: bool = False, key_ex: bool = False, doc2vec: bool = False, classification: bool = False):
        result_scores = []
        if lda:
            result_scores = self.__lda_pipeline()
        return result_scores

    def __lda_pipeline(self):
        model = pickle.load(open("models/LDA/LdaModel_ntopics_3_freq_sd_arxiv.bin", "rb"))
        vectorizer = pickle.load(open("models/LDA/vectorizer.bin", "rb"))
        preprocessor = Preprocessor(self.corpora.data)
        docs = preprocessor.lemmatize(remove_stops=True, remove_punct_and_nums=True)
        scores_per_sentence = list()
        for doc in docs:
            input_sentence = vectorizer.transform([" ".join(doc)])
            topic_scores = model.transform(input_sentence)
            scores_per_sentence.append(list(np.array(topic_scores).flatten()))
        return scores_per_sentence

    # TODO __merge_predictions() need to be implemented
    """
    def __merge_predictions(self, predictions: list) -> Prediction:
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        pass
        # return Prediction()
    """
