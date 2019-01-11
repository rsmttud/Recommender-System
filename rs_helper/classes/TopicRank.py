from rs_helper.classes.KeywordExtractor import KeywordExtractor
from rs_helper.classes import Topic
import string
from nltk.corpus import stopwords
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

    def extract_keywords(self, *kwargs):
        result = {}
        candidates = list()
        for i, p in enumerate(self.paths):
            topic_rank = pke.unsupervised.TopicRank()
            topic_rank.load_document(input=p, language="en")
            topic_rank.candidate_selection(pos=self.pos, stoplist=self.stoplist)
            topic_rank.candidate_weighting(threshold=0.74, method='average', heuristic="frequent")
            topic_rank_keyphrases = topic_rank.get_n_best(n=self.top_n)
            candidates.append(topic_rank.candidates)
            topic = self.__generate_topic(topic_rank_keyphrases, self.labels[i])
            result.update({self.labels[i]: topic})
        self.candidates = candidates
        return result

    def __generate_topic(self, token_rank_dict, label: str):
        topic = Topic(class_name=label)
        for w, v in token_rank_dict:
            topic.add_keyword(keyword=w.split(" "), rank=v, algorithm=self.class_name)
        return topic
