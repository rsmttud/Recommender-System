from rs_helper.classes.KeywordExtractor import KeywordExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from rs_helper.classes import Topic


class TFIDF(KeywordExtractor):

    def __init__(self, documents: list, labels: list, top_n: int):
        super().__init__()
        self.docs = documents
        self.labels = labels
        self.top_n = top_n

    def extract_keywords(self, *kwargs):
        vectorizer = TfidfVectorizer(lowercase=True, preprocessor=None, tokenizer=None,
                                     analyzer="word", stop_words=None, ngram_range=(1, 3))
        scores = vectorizer.fit_transform(self.docs)
        index_value = {i[1]: i[0] for i in vectorizer.vocabulary_.items()}
        result = dict()
        for i, row in enumerate(scores):
            values = {index_value[column]: value for (column, value) in zip(row.indices, row.data)}
            values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
            topic = self.__generate_topic(values, self.labels[i])
            result.update({self.labels[i]: topic})
        return result

    def __generate_topic(self, token_rank_dict, label: str):
        topic = Topic(class_name=label)
        for el, va in token_rank_dict:
            topic.add_keyword(el, va, self.class_name)
        return topic