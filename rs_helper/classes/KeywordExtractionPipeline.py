from rs_helper.classes.Model import Model
from rs_helper.classes.FastTextEncoder import FastTextEncoder
from rs_helper.classes import Topic
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from rs_helper.classes.Prediction import Prediction
from rs_helper.classes.Preprocessor import Preprocessor
from nltk import bigrams, trigrams
from functools import reduce
import operator
from typing import *
import numpy as np
import os


class KeywordExtractionPipeline(Model):

    def __init__(self, path_to_topics: str, cuttoff_value: int = 75):
        """
        :param path_to_topics: String: Path to topic directory.
        All topics available in specified path will be loaded. Restricted to all files with ".joblib" ending.
        Convention for topic files: <label>.joblib
        """
        super().__init__(path_to_model=path_to_topics)
        self.topics = list()
        self.cutoff_value = cuttoff_value
        self.topic_keywords = list()
        self.labels = list()
        self.base_vectors = None

    def initialize(self):
        """
        :return: void
        Initialization of Keyword Classifier. Loads topics from supplied path and created base vectors
        for similarity comparison.
        """
        for f in [x for x in os.listdir(self.path) if x.endswith(".joblib")]:
            label, _ = f.split(".")
            self.labels.append(label)
            topic: Topic = load(os.path.join(self.path, f))
            self.topics.append(topic)
            keywords = topic.get_keyword_names()
            reduced = self.__clean(keywords)
            self.topic_keywords.append(reduced[:self.cutoff_value])

        self.base_vectors = FastTextEncoder(documents=self.topic_keywords, sentence_level=True).vectorize()

    def __clean(self, kw_list: List[List[str]]):
        """
        :param kw_list: List(List(String)): List of keywords of a topic
        :return: List(String): Reduced list with joined n-grams
        """
        return [" ".join(kw) for kw in kw_list]

    def __preprocess(self, text: str) -> List[List[str]]:
        """
        :param text: String: Text to preprocess
        :return: List(List(String)): Preprocessed data per sentence
        Method to preprocess the data with Preprocessor
        """
        pp = Preprocessor(data=text)
        return pp.transform(remove_nums=True, remove_punct=True, synsets=False)

    def predict(self, text: str) -> Prediction:
        """
        :param text: String: Input text to classify
        :return: Prediction
        Method to make nearest neigbor classification based on cosine similarity in vector space
        """
        if len(self.topics) == 0:
            raise IOError("Model needs to be initialized first. Call model.initialize().")
        tokens = self.__preprocess(text)
        tokens = self.__extend(tokens)
        in_vecs = FastTextEncoder(documents=tokens, sentence_level=True).vectorize()
        sims = self.__calculate_similarities(vectors=in_vecs)
        pred = Prediction(values=list(sims), classes=self.labels)
        return pred

    def __calculate_similarities(self, vectors):
        sims = np.zeros(len(self.labels))
        for vec in vectors:
            s_sims = [cosine_similarity([vec], [b]) for b in self.base_vectors]
            sims += np.array([s[0][0] for s in s_sims])
        # TODO On whole text or only on sentence?
        # vec = sum([np.array(v) for v in in_vecs])/len(in_vecs)
        # sims = [cosine_similarity([vec], [b]) for b in self.base_vectors]
        # sims = [s[0][0] for s in sims]
        return list(sims)

    def __extend(self, tokens: List[List[str]]) -> List[List[str]]:
        """
        :param tokens: List(List(String)): List of tokenized sentences
        :return: List(List(String)): List of extended tokenized sentences
        Method extends the supplied list of lists of tokens with every possible bi- and trigrams
        """
        extended_tokens = list()
        for sent in tokens:
            bigram_tokens = [' '.join(t) for t in bigrams(sent)]
            trigram_tokens: list = [''.join(t) for t in trigrams(sent)]
            all_tokens = reduce(operator.concat, [sent, bigram_tokens, trigram_tokens])
            extended_tokens.append(all_tokens)
        return extended_tokens

    def normalize_result(self, prediction: Prediction):
        pass
