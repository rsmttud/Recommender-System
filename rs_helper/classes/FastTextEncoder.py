from rs_helper.classes.Vectorizer import Vectorizer
from gensim.models import FastText
import pickle


class FastTextEncoder(Vectorizer):

    def __init__(self, documents: list, sentence_level: bool = False):
        """
        :param documents: List(List(String)) : List of tokenized sentences
        :param sentence_level: Boolean : Merge every sentence to one vector
        """
        super().__init__()
        if not isinstance(documents, list) or not isinstance(documents[0], list):
            raise ValueError("Parameter documentes needs to be of type List(List(String))")
        self.documents = documents
        self.sentence_level = sentence_level
        self.model = None
        self.initialize()

    def initialize(self):
        model = pickle.load(open("notebooks/model_trainings/FastText/models/ft_model_15000.pkl", "rb"))
        self.model = model

    def vectorize(self, **kwargs):
        vectorized = list()
        for doc in self.documents:
            vectors = []
            for token in doc:
                try:
                    vectors.append(self.model.wv.__getitem__(token))
                except:
                    pass
            if self.sentence_level:
                vectorized.append(sum(vectors) / len(vectors))
            else:
                vectorized.append(vectors)
        return vectorized
