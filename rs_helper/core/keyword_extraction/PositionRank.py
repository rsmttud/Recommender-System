from rs_helper.core.keyword_extraction.KeywordExtractor import KeywordExtractor
from rs_helper.core.Topic import Topic
from rs_helper.core.Keyword import Keyword
import pke


class PositionRank(KeywordExtractor):

    def __init__(self, paths_to_files: list, labels: list, top_n: int):
        super().__init__()
        self.paths = paths_to_files
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.candidates = None
        self.labels = labels
        self.top_n = top_n

    # TODO need to be adjusted to Keyword Class
    def extract_keywords(self, *kwargs):
        result = {}
        candidates = list()
        for i, p in enumerate(self.paths):
            position_rank = pke.unsupervised.PositionRank()
            position_rank.load_document(input=p, language="en", normalization=None)
            position_rank.candidate_selection(maximum_word_number=1)
            position_rank.candidate_weighting(window=5, pos=self.pos, normalized=True)
            position_rank_keyphrases = position_rank.get_n_best(n=self.top_n)
            candidates.append(position_rank.candidates)
            topic = self.generate_topic(position_rank_keyphrases, self.labels[i])
            result.update({self.labels[i]: topic})
        self.candidates = candidates
        #keywords = Keyword()
        return result

    def generate_topic(self, token_rank_dict, label: str):
        topic = Topic(class_name=label)
        for w, v in token_rank_dict:
            topic.add_keyword(keyword=w.split(" "), rank=v, algorithm=self.class_name)
        return topic
