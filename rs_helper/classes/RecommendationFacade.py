from rs_helper.classes import Corpora
from rs_helper.classes import Prediction
from rs_helper.classes.EmbeddingClassificationPipeline import EmbeddingClassificationPipeline
from rs_helper.classes.LatentDirichletAllocation import LatentDirichletAllocation
from rs_helper.classes.KeywordExtractionPipeline import KeywordExtractionPipeline
from rs_helper.classes.OneToOneGRU import OneToOneGRU
from rs_helper.classes.SVC import SVC
import pickle


class RecommendationFacade:

    def __init__(self, path_to_files: str):
        self.corpora = Corpora(path=path_to_files)

    def run(self, lda: bool = False, key_ex: bool = False,
            doc2vec: bool = False, classification: bool = False,
            svc_classification: bool = False, gru_oto: bool = False):
        """
        Function should manage the starting of pipelines according to input
        :param lda: boolean to start or not start LDA pipeline
        :param key_ex: boolean to start or not start Keyword Extraction pipeline
        :param doc2vec: boolean to start or not start Doc2Vec pipeline
        :param classification: boolean to start or not start ML Classification pipeline
        :param svc_classification: boolean to start or not start ML SVC Classification pipeline
        :param gru_oto: boolean to start the GRU One To One Classification pipeline
        :return: Object of type Prediction (merged prediction of all pipelines)
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

    def __lda_pipeline(self):
        path_model = "models/LDA/LdaModel_3_freq_clean.bin"
        path_vectorizer = "models/LDA/vectorizer_3_freq_clean.bin"
        model = LatentDirichletAllocation(path_to_model=path_model, path_to_vectorizer=path_vectorizer)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction

    def __classification_embedding(self) -> Prediction:
        path_classifier = "./models/classifier/daniel_0712/dnn_0712"
        path_embedding = "models/trained_models/daniel_0712/USE_DAN/use_081218"  # Still useless
        model = EmbeddingClassificationPipeline(path_to_embedding_model=path_embedding, path_to_model=path_classifier)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction

    def __svc_classification(self):
        path_classifier = "models/classifier/SVC/SVC_tfidf_clean_.pkl"
        path_vectorizer = "models/classifier/SVC/TFIDF_SVC_tfidf_clean_.pkl"
        model = SVC(path_to_model=path_classifier, path_to_vectorizer=path_vectorizer)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction

    def __key_ex_pipeline(self):
        path_topics = "data/topics/"
        model = KeywordExtractionPipeline(path_to_topics=path_topics)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction

    def __gru_oto_classification(self):
        path_model = "models/classifier/GRU_OtO/gru_one_to_one_equal_sets_200_units_0.2_dropout_20_epochs.yaml"
        path_weights = "models/classifier/GRU_OtO/gru_one_to_one_equal_sets_weights_200_units_0.2_dropout_20_epochs.h5"
        path_encoder = "notebooks/model_trainings/FastText/models/ft_model_15000.pkl"
        model = OneToOneGRU(path_to_model=path_model,
                            path_to_weights=path_weights,
                            path_to_encoder=path_encoder)
        model.initialize()
        predictions = model.predict(self.corpora.data)
        return predictions

    def __merge_predictions(self, predictions: list) -> Prediction:
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        return Prediction(["dummy"], ["dummy"])
