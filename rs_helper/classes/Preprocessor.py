import spacy
import re


class Preprocessor:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def word_tokenize(self, sentence: str) -> list:
        doc = self.nlp(sentence)
        return [token.text.lower() for token in doc]

    def sent_tokenize(self, document: str) -> list:
        pass

    def remove_stopwords(self, tokens: list) -> list:
        pass

    def lemmatize(self, tokens: list) -> list:
        pass

    def remove_punct_and_nums(self, tokens: list) -> list:
        pass
