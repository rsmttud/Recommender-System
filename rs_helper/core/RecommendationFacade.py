from rs_helper.core.Corpora import Corpora
from rs_helper.core.Prediction import Prediction

from rs_helper.core.model.RNNTypedClassifier import RNNTypedClassifier
from rs_helper.core.model.LatentDirichletAllocation import LatentDirichletAllocation
from rs_helper.core.distributed_models.DAN import DAN
from rs_helper.core.distributed_models.FastTextWrapper import FastTextWrapper
from rs_helper.core.model.SVCModel import SVCModel
from typing import List


# from rs_helper.cor.EmbeddingClassificationPipeline import EmbeddingClassificationPipeline
# from rs_helper.classes.KeywordExtractionPipeline import KeywordExtractionPipeline


class RecommendationFacade:
    """
    General Class to handle to predictions. Actual implementation of the Facade-Pattern.
    """
    def __init__(self, path_to_files: str):
        """
        Constructor of Facade

        :param path_to_files: Directory where all files are stored. They will be loaded in Corpora object
        :type path_to_files: str
        """
        self.corpora = Corpora(path=path_to_files)

    def run(self, lda: bool = False, key_ex: bool = False,
            doc2vec: bool = False, classification: bool = False,
            svc_classification: bool = False, gru_oto: bool = False):
        """
        Function should manage the starting of pipelines according to input

        :param lda: boolean to start or not start LDA pipeline
        :type lda: bool
        :param key_ex: boolean to start or not start Keyword Extraction pipeline
        :type key_ex: bool
        :param doc2vec: boolean to start or not start Doc2Vec pipeline
        :type doc2vec: bool
        :param classification: boolean to start or not start ML Classification pipeline
        :type classification: bool
        :param svc_classification: boolean to start or not start ML SVC Classification pipeline
        :type svc_classification: bool
        :param gru_oto: boolean to start the GRU One To One Classification pipeline
        :type gru_oto: bool

        :return: Merged prediction of all pipelines
        :rtype: Prediction
        """
        if lda:
            return self.__lda_pipeline()
        if key_ex:
            return self.__key_ex_pipeline()
        if classification:
            return self.__classification_embedding()
        if svc_classification:
            return self.__svc_classification()
        if gru_oto:
            return self.__gru_oto_classification()

    # TODO they probably all need an adjustment
    def __lda_pipeline(self):
        path_model = "models/LDA/LdaModel_3_freq_clean.bin"
        path_vectorizer = "models/LDA/vectorizer_3_freq_clean.bin"
        model = LatentDirichletAllocation(path_to_model=path_model, path_to_vectorizer=path_vectorizer)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction

    def __classification_embedding(self) -> Prediction:
        ft_model = FastTextWrapper("models/FastText/1/model.joblib")
        dan = DAN(ft_model, "models/DANs/1/frozen_graph.pb")
        _svc = SVCModel("models/SVC/1/model.joblib", dan)
        prediction = _svc.predict(self.corpora.data)
        prediction.scale_log()
        prediction.round_values()
        #prediction.values = [x*100 for x in prediction.values]
        return prediction

    def __svc_classification(self):
        """
        path_classifier = "models/classifier/SVC/SVC_tfidf_clean_.pkl"
        path_vectorizer = "models/classifier/SVC/TFIDF_SVC_tfidf_clean_.pkl"
        model = SVC(path_to_model=path_classifier, path_to_vectorizer=path_vectorizer)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        """

        return None

    def __key_ex_pipeline(self):
        pass
        # Pipelines don't exist anymore
        """
        path_topics = "data/topics/"
        model = KeywordExtractionPipeline(path_to_topics=path_topics)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction
        """

    def __gru_oto_classification(self):
        path_model = "models/classifier/GRU_OtO/gru_one_to_one_equal_sets_200_units_0.2_dropout_20_epochs.yaml"
        path_weights = "models/classifier/GRU_OtO/gru_one_to_one_equal_sets_weights_200_units_0.2_dropout_20_epochs.h5"
        path_encoder = "notebooks/model_trainings/FastText/models/ft_model_15000.pkl"
        model = RNNTypedClassifier(path_to_model=path_model,
                                   path_to_weights=path_weights,
                                   path_to_encoder=path_encoder)
        model.initialize()
        predictions = model.predict(self.corpora.data)
        return predictions

    def __merge_predictions(self, predictions: List[Prediction]) -> Prediction:
        """
        Performs Ensemble Learning with Ensemble class

        :param predictions: list of all predictions
        :type predictions: list(Prediction)

        :return: Final prediction
        :rtype: Prediction
        """
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        return Prediction(["dummy"], ["dummy"])
