from rs_helper.classes import Corpora
from rs_helper.classes import Vectorizer
from rs_helper.classes import Prediction
from rs_helper.classes import Preprocessor
from rs_helper.classes.EmbeddingClassificationPipeline import EmbeddingClassificationPipeline


class RecommendationFacade:

    def __init__(self, path_to_file: str):
        self.corpora = Corpora(path=path_to_file)

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
        if classification:
            return self.__classification_embedding()

    def __lda_pipeline(self):
        pass

    def __classification_embedding(self):
        path_classifier = "models/trained_models/daniel_0712/dnn_0712"
        path_embedding = "models/trained_models/daniel_0712/USE_DAN/use_081218"
        model = EmbeddingClassificationPipeline(path_to_embedding_model=path_embedding, path_to_model=path_classifier)
        model.initialize()
        prediction = model.predict(self.corpora.data)
        return prediction

    def __key_ex_pipeline(self):
        pass

    def __merge_predictions(self, predictions: list) -> Prediction:
        if not isinstance(predictions[0], Prediction):
            raise ValueError("Parameter predictions must be of type List(Prediction)")
        return Prediction(["dummy"], ["dummy"])
