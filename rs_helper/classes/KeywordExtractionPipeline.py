from rs_helper.classes.Model import Model
from rs_helper.classes.OneHotEncoder import OneHotEncoder
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from rs_helper.classes.Prediction import Prediction
from rs_helper.classes.Preprocessor import Preprocessor
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize
from functools import reduce
import operator
import os


class KeywordExtractionPipeline(Model):

    def __init__(self, path_to_model: str, path_to_vocab: str):
        super().__init__(path_to_model)
        self.topics = None
        self.vocab = pickle.load(open(path_to_vocab, "rb"))
        self.encoder = None
        self.preprocessor = None

    def initialize(self):
        self.encoder = OneHotEncoder(vocab=self.vocab)
        topics = dict()
        for file in os.listdir(self.path):
            if file.endswith(".topic"):
                topic = pickle.load(open(os.path.join(self.path, file), "rb"))
                topic_words = ["_".join(kw).strip() for kw in topic.get_keyword_names()]
                topics[file.split(".")[0]] = self.encoder.vectorize(text=topic.get_keyword_names())
        self.topics = topics

    def predict(self, text: str):
        preprocessor = Preprocessor(data=text)
        transformed = reduce(operator.concat, preprocessor.transform())
        extended_input_text = self.__expand(transformed)
        input_text = self.encoder.vectorize(text=extended_input_text)
        classes, values = list(), list()
        for t in self.topics:
            print(t)
            sim = cosine_similarity(input_text.reshape(1, -1), self.topics[t].reshape(1, -1)) * 100
            print(sim)
            classes.append(t)
            values.append(sim)
        print("___")
        pred = Prediction(classes=classes, values=values)
        return pred

    def __expand(self, text: list):
        unique_tokens = list(set(text))
        bigram_tokens = ['_'.join(token) for token in bigrams(text)]
        trigram_tokens: list = ['_'.join(token) for token in trigrams(text)]
        all_tokens = reduce(operator.concat, [unique_tokens, bigram_tokens, trigram_tokens])
        print(all_tokens)
        return all_tokens

    def normalize_result(self, prediction: Prediction):
        pass