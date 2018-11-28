import spacy


class Preprocessor:

    def __init__(self, data: str):
        self.nlp = spacy.load('en_core_web_sm')
        self.doc = self.nlp(data)

    def word_tokenize(self) -> list:
        return [token.lower_ for token in self.doc]

    def sent_tokenize(self) -> list:
        return [sent for sent in self.doc.sents]

    def remove_stopwords(self) -> list:
        return [token for token in self.doc if not token.is_stop]

    def lemmatize(self) -> list:
        return [token.lemma_ for token in self.doc]

    def remove_punct_and_nums(self) -> list:
        return [token for token in self.doc if token.is_alpha and not token.is_punct]

    def to_text(self) -> list:
        return [el.lower_ for el in self.doc]
