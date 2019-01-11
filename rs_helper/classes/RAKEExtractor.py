import RAKE
import os
from nltk.corpus import stopwords
from rs_helper.classes import KeywordExtractor


class RAKEExtractor(KeywordExtractor):
    def __init__(self, documents: list, labels: list):
        super().__init__()
        self.data = documents
        self.model = RAKE.Rake(stopwords.words("english"))
        self.labels = labels

    def extract_keywords(self, *kwargs):
        for j, d in enumerate(self.data):
            print(self.labels[j])
            keywords = self.model.run(d, minCharacters=4, maxWords=2, minFrequency=2)
            for i, word in enumerate(keywords):
                print(word)
                if i == 20:
                    break
        return keywords

    def __generate_topic(self, token_rank_dict, label: str):
        pass