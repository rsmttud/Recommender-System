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

    def lemmatize(self, remove_stops: bool = True, remove_punct_and_nums: bool = True) -> list:
        lemmatized = list()
        for sent in self.doc.sents:
            if not remove_punct_and_nums and not remove_stops:
                lemmatized.append([token.lemma_ for token in sent])
            if remove_stops and not remove_punct_and_nums:
                lemmatized.append([token.lemma_ for token in sent if not token.is_stop])
            if remove_punct_and_nums and not remove_stops:
                lemmatized.append([token.lemma_ for token in sent if token.is_alpha and not token.is_punct])
            if remove_stops and remove_punct_and_nums:
                lemmatized.append([token.lemma_ for token in sent if token.is_alpha and not token.is_stop])
        return lemmatized

    def remove_double_spaces(self, strings):
        for sent in strings:
            sent.replace("  ", " ")
            if sent.find("  ") != -1:
                strings.remove(sent)
        return strings

    def remove_punct_and_nums(self) -> list:
        return [token for token in self.doc if token.is_alpha and not token.is_punct]

    def to_text(self) -> list:
        return [el.lower_ for el in self.doc]

    def preprocess(self, text: str, lowercase: bool = True, remove_punct_and_nums: bool = True,
                   lemmatize: bool = True, remove_stopwords: bool = True):
        if lowercase:
            text = text.lower()
