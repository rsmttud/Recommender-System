from nltk import bigrams, trigrams
import math
from tqdm import tqdm
from functools import reduce
import operator

from rs_helper.core.keyword_extraction.KeywordExtractor import KeywordExtractor
from rs_helper.core.Topic import Topic
from rs_helper.core import Keyword


class TFIGM(KeywordExtractor):

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
        self.lambda_value = 7

    def __compute_tf(self, word: str, document):
        """
        Computes the term frequency of a word in a given document

        :param word: The word
        :type word: str
        :param document: The document
        :type document: str

        :return: The frequency of a word
        :rtype: int
        """
        return document.count(word)

    def __num_words(self, document: list):
        """
        Compute the length of a document

        :param document: the tokenized document
        :type document: list(str)

        :return: The document length
        :rtype: int
        """
        self.num_words = len(document)
        return self.num_words

    def __compute_df(self, key: str):
        """
        Compute document frequency for a word

        :param key: word
        :type key: str

        :return: The count of the word across corpus
        :rtype: int
        """
        count = 0
        for doc in self.joined_docs:
            if key in doc:
                count += 1
        return count

    def __compute_idf(self, key: str):
        """
        Computes the inverse document frequency

        :param key: word
        :type key: str

        :return: Inverse document frequency
        :rtype: float
        """
        count = self.__compute_df(key)
        return math.log((self.num_docs/count))

    def term_frequency(self, word: str, document: list) -> float:
        """
        Calculate the term frequency of a word

        :param word: word
        :type word: str
        :param document: list of documents
        :type document: list(str)

        :return: the term frequency
        :rtype: float
        """
        return float(self.__compute_tf(word, document) / self.__num_words(document))

    def frequency_class_distribution(self, word: str, documents: list):
        """
        Compute the frequncy class distribution of a word.

        :param word: word
        :type word: str
        :param documents: list of documents
        :type documents: list(str)

        :return: Sorted list of term frequencies over documents
        :rtype: list(str)
        """

        frequencies = [self.__compute_tf(word, d) for d in documents]
        return sorted(frequencies, reverse=True)

    def __get_vocab(self):
        """
        Generate basic vocab containing all uni-, bi- and trigrams of the documents.

        :return: list of tokens
        :rtype: list
        """
        vocab = list()
        for doc in self.data:
            unique_tokens = list(set(doc))
            bigram_tokens = bigrams(doc)  # Returns list of tupels
            bigram_tokens = [' '.join(token) for token in bigram_tokens]
            trigram_tokens: list = trigrams(doc)  # Returns list of tupels
            trigram_tokens: list = [' '.join(token) for token in trigram_tokens]
            all_tokens = reduce(operator.concat, [unique_tokens, bigram_tokens, trigram_tokens])
            vocab.append(all_tokens)
            del unique_tokens, bigram_tokens, trigram_tokens, all_tokens
        return vocab

    def extract_keywords(self):
        results = dict()
        vocab = self.__get_vocab()
        # pickle.dump(reduce(operator.concat, vocab), open("data/topics/tfigm_vocab.vocab", "wb"))
        for i, doc in enumerate(vocab):
            word_tfigm_dict = dict()
            for token in tqdm(doc):
                term_freq = self.term_frequency(token, doc)
                term_class_distribution = self.frequency_class_distribution(token, vocab)
                weighted_sum_of_frequencies = sum([value * (index + 1)
                                                   for index, value in enumerate(term_class_distribution)])

                tf_igm = term_freq * (1 + self.lambda_value * (term_class_distribution[0] / weighted_sum_of_frequencies))
                word_tfigm_dict[token] = tf_igm
            sorted_dict = sorted(word_tfigm_dict.items(), key=lambda x: x[1], reverse=True)
            # print(sorted_dict)
            topic = self.generate_topic(sorted_dict[:self.top_n],
                                        label=self.labels[i])
            results.update({self.labels[i]: topic})
        return results

    def generate_topic(self, tdidf_values, label: str) -> Topic:
        topic = Topic(class_name=label)
        for key, value in tdidf_values:
            topic.add_keyword(keyword=key, rank=value, algorithm=self.class_name)
        return topic


