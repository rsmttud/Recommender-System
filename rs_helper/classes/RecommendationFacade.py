from rs_helper.classes import Corpora
from rs_helper.classes import Vectorizer
from rs_helper.classes import Prediction
from rs_helper.classes.Preprocessor import Preprocessor
import pickle
import numpy as np



class RecommendationFacade:

    def __init__(self, path_to_file: str):
        corpora = Corpora(path=path_to_file)

    def run(self, lda: bool = False, key_ex: bool = False, doc2vec: bool = False, classification: bool = False):
        """
        Function should manage the starting of pipelines according to input
        :param lda: boolean to start or not start LDA pipeline
        :param key_ex: boolean to start or not start Keyword Extraction pipeline
        :param doc2vec: boolean to start or not start Doc2Vec pipeline
        :param classification: boolean to start or not start ML Classification pipeline
        :return: Object of type Prediction (merged prediction of all pipelines)
        """
        if lda:
            self.__lda_pipeline()
        if key_ex:
            self.__key_ex_pipeline()

    def __lda_pipeline(self):
        model = pickle.load(open("models/LDA/LdaModel_ntopics_3_freq.bin", "rb"))
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
        return Prediction(["dummy"], ["dummy"])
