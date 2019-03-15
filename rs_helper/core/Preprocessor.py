from sklearn.feature_extraction.text import CountVectorizer
from typing import List
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


class LemmaTokenizer(object):
    """
    Class to perform lemmatization in preprocessing
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, text):
        return [self.wnl.lemmatize(t) for t in word_tokenize(text)]


class Preprocessor:
    """
    Class to preprocess textual data
    """
    def __init__(self, data: str) -> None:
        """
        Constructor of the Preprocessor class.
        Sklean CountVectorizer will be initialized.

        :param data: The data to preprocess
        :type data: str
        """
        self.data = [s for s in sent_tokenize(data)]
        self.vectorizer = CountVectorizer(stop_words=list(stopwords.words("english")),
                                          min_df=1, decode_error="ignore",
                                          strip_accents="ascii", ngram_range=(1, 1),
                                          tokenizer=LemmaTokenizer())
        self.vectorizer.fit(self.data)
        self.analyzer = self.vectorizer.build_analyzer()

    def lemmatize(self) -> None:
        """
        Lemmatize the text data. Results will be stored in class attribute data.

        :return: None
        """
        self.data = [self.analyzer(s) for s in self.data]

    def remove_puctuation(self) -> None:
        """
        Remove punctuation from data. Results will be stored in class attribute data.

        :return: None
        """
        self.data = [[x for x in s if x not in string.punctuation] for s in self.data]

    def remove_numbers(self) -> None:
        """
        Remove numbers from data. Results will be stored in class attribute data.

        :return:
        """
        self.data = [[x for x in s if re.match("[A-Za-z]+", x) is not None] for s in self.data]

    def transform(self, remove_nums: bool = True, remove_punct: bool = True) -> List[List[str]]:
        """
        General handler to transform textual data.

        :param remove_nums: Remove numbers?
        :type remove_nums: bool
        :param remove_punct: Remove Punctuation?
        :type remove_punct: bool

        :return: preprocessed data
        :rtype: list
        """
        self.lemmatize()
        if remove_nums and not remove_punct:
            self.remove_numbers()
        if remove_punct and not remove_nums:
            self.remove_puctuation()
        if remove_nums and remove_punct:
            self.remove_puctuation()
            self.remove_numbers()
        return self.data
