from rs_helper.core.keyword_extraction.KeywordExtractor import KeywordExtractor
from rs_helper.core.Topic import Topic
import string
from nltk.corpus import stopwords
from typing import *
import pke


class TopicRank(KeywordExtractor):

    def __init__(self, paths_to_files: list, labels: list, top_n: int):
        super().__init__()
        self.paths = paths_to_files
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += stopwords.words('english')
        self.candidates = None
        self.labels = labels
        self.top_n = top_n

    def extract_keywords(self, *kwargs) -> Dict[str, Topic]:
        result = {}
        candidates = list()
        for i, p in enumerate(self.paths):
            topic_rank = pke.unsupervised.TopicRank()
            topic_rank.load_document(input=p, language="en")
            topic_rank.candidate_selection(pos=self.pos, stoplist=self.stoplist)
            topic_rank.candidate_weighting(threshold=0.74, method='average', heuristic="frequent")
            topic_rank_keyphrases = topic_rank.get_n_best(n=self.top_n)
            candidates.append(topic_rank.candidates)
            topic = self.generate_topic(topic_rank_keyphrases, self.labels[i])
            result.update({self.labels[i]: topic})
        self.candidates = candidates
        return result

    def generate_topic(self, token_rank_dict: List[Tuple[str, float]], label: str) -> Topic:
        topic = Topic(class_name=label)
        for w, v in token_rank_dict:
            topic.add_keyword(keyword=w.split(" "), rank=v, algorithm=self.class_name)
        return topic
