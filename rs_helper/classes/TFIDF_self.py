from rs_helper.classes.KeywordExtractor import KeywordExtractor
from rs_helper.classes import Topic
from rs_helper.classes import Corpora
from nltk import bigrams, trigrams
import math
import numpy as np
from functools import reduce
import operator


class TFIDF(KeywordExtractor):

    def __init__(self, data: list, labels: list, top_n: int = 100):
        """
        :param data: List(List(String)) (List of tokenized documents)
        :param labels: List(String) (List of labels associated with documents)
        """
        if not isinstance(data, list) or not isinstance(labels, list):
            raise ValueError("Parameters data and labels needs to be of type List.")
        if not len(data) == len(labels):
            raise ValueError("The provided list of documents and labels need to have same length.")
        super().__init__()
        self.key_score_dict = dict()
        self.data = data
        self.labels = labels
        self.top_n = top_n
        self.num_words = 0
        self.num_docs = len(data)
        self.joined_docs = [" ".join(doc) for doc in data]

    def __compute_tf(self, word: str, document: str):
        return document.count(word)

    def num_words(self, document: list):
        self.num_words = len(document)
        return self.num_words

    def __compute_df(self, key: str):
        count = 0
        for doc in self.joined_docs:
            if key in doc:
                count += 1
        return count

    def __compute_idf(self, key: str):
        count = self.__compute_df(key)
        return math.log((self.num_docs/count))

    def __get_all_tokens(self, doc: list):
        unique_tokens = list(set(doc))
        bigram_tokens = bigrams(doc)  # Returns list of tupels
        bigram_tokens = [' '.join(token) for token in bigram_tokens]
        trigram_tokens: list = trigrams(doc)  # Returns list of tupels
        trigram_tokens: list = [' '.join(token) for token in trigram_tokens]
        all_tokens = reduce(operator.concat, [unique_tokens, bigram_tokens, trigram_tokens])
        return all_tokens

    def extract_keywords(self) -> dict:
        results = dict()
        for i, doc in enumerate(self.data):
            word_tfidf_dict = dict()
            all_tokens = self.__get_all_tokens(doc=doc)
            print(len(all_tokens))
            for token in all_tokens:
                tfidf = self.__compute_tf(token, doc) * self.__compute_idf(token)
                word_tfidf_dict[token] = tfidf
            sorted_dict = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
            # print(sorted_dict)
            topic = self.__generate_topic(sorted_dict[:self.top_n],
                                          label=self.labels[i])
            results.update({self.labels[i]: topic})
        return results

    def __generate_topic(self, tdidf_values, label: str) -> Topic:
        topic = Topic(class_name=label)
        for key, value in tdidf_values:
            topic.add_keyword(keyword=key, rank=value, algorithm=self.class_name)
        return topic


