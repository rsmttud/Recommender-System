from sklearn.feature_extraction.text import CountVectorizer
import os
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, text):
        return [self.wnl.lemmatize(t) for t in word_tokenize(text)]


class Preprocessor:

    def __init__(self, data: str):
        self.data = [s for s in sent_tokenize(data)]
        self.vectorizer = CountVectorizer(stop_words=list(stopwords.words("english")),
                                          min_df=1, decode_error="ignore",
                                          strip_accents="ascii", ngram_range=(1, 1),
                                          tokenizer=LemmaTokenizer())
        self.vectorizer.fit(self.data)
        self.analyzer = self.vectorizer.build_analyzer()

    def lemmatize(self):
        self.data = [self.analyzer(s) for s in self.data]

    def remove_puctuation(self):
        self.data = [[x for x in s if x not in string.punctuation] for s in self.data]

    def remove_numbers(self):
        self.data = [[x for x in s if re.match("[A-Za-z]+", x) is not None] for s in self.data]

    def transform(self, remove_nums: bool = True, remove_punct: bool = True):
        self.lemmatize()
        if remove_nums and not remove_punct:
            self.remove_numbers()
        if remove_punct and not remove_nums:
            self.remove_puctuation()
        if remove_nums and remove_punct:
            self.remove_puctuation()
            self.remove_numbers()
        return self.data
