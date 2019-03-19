from rs_helper.core import Prediction
from rs_helper.core.model import Model
from rs_helper.core.distributed_models import EmbeddingModel, DAN, FastTextWrapper
from joblib import load
from typing import Dict
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TopicKNNModel(Model):
    """
    Class for nearest neighbor classification in vector space against topic embeddings
    """

    def __init__(self, path_to_topic: str, embedding_model: EmbeddingModel) -> None:
        """
        Constructor of Topic KNN classifier.

        :param path_to_topic: The file in which the topics are stored
        :type path_to_topic: str
        :param embedding_model: The embedding model to generate the vector representations
        :type embedding_model: EmbeddingModel
        """
        if not isinstance(embedding_model, DAN) or not isinstance(embedding_model, FastTextWrapper):
            raise ValueError("Embedding model must be of type DAN or FastTextWrapper")

        super().__init__(path_to_topic)
        self.path = path_to_topic
        self.topics = None
        self.vectors = None
        self.embedding_model = embedding_model
        self.initialize()

    def initialize(self) -> None:
        """
        Initializes the model by loading all topics and building up their vector representations

        :return: None
        """
        topic_dict = load(self.path)
        self.topics = topic_dict
        vector_dict = self.topic2vec()
        self.vectors = vector_dict

    def topic2vec(self) -> Dict[str, np.ndarray]:
        """
        Creates topic embeddings based on the topic objects using the supplied embedding model

        :return: Dict with embedding per class
        :rtype: dict(str, np.ndarray)
        """
        return_dict = dict()
        for cl in self.topics:
            keywords = self.topics[cl].get_keyword_names()
            tokens = list()
            for x in keywords:
                if isinstance(x, list):
                    k = " ".join(x)
                    tokens.append(k)
                else:
                    tokens.append(x)

            if isinstance(self.embedding_model, FastTextWrapper):
                topic_embedding = self.embedding_model.inference(tokens, sentence_level=True)
            else:
                topic_embedding = self.embedding_model.inference(tokens)
            return_dict[cl] = topic_embedding
        return return_dict

    def predict(self, text: str) -> Prediction:
        """
        Predicts the given input text according to cosine similarity and returns this in standardized Prediction object.

        :param text: Text to predict.
        :type text: str

        :return: The final prediction
        :rtype: Prediction
        """
        if self.vectors is None:
            raise IOError("Model needs to initialized first. Please call KNN.initialize() first.")
        input_embedding = self.embedding_model.inference(word_tokenize(text)) if isinstance(self.embedding_model, DAN) \
            else self.embedding_model.inference(word_tokenize(text), sentence_level=True)
        similarities = list()
        for cl in self.vectors:
            emb = self.vectors[cl]
            similarities.append(cosine_similarity([input_embedding], [emb])[0])
        prediction = Prediction(classes=list(self.vectors.keys()), values=similarities)
        return prediction

    def normalize_result(self, prediction: Prediction) -> Prediction:
        pass
