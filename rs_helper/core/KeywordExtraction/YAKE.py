import string
from nltk.corpus import stopwords
import pke
from rs_helper.core.KeywordExtraction.KeywordExtractor import KeywordExtractor
from rs_helper.core.Topic import Topic
from rs_helper.core.Keyword import Keyword


class YAKE(KeywordExtractor):

    def __init__(self, paths_to_files: list, labels: list, top_n: int, threshold: float = 0.8):
        super().__init__()
        self.paths = paths_to_files
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += stopwords.words('english')
        self.threshold = threshold
        self.candidates = None
        self.labels = labels
        self.top_n = top_n

    # TODO adjust to Keywords
    def extract_keywords(self, *kwargs):
        result = {}
        candidates = list()
        for i, p in enumerate(self.paths):
            yake = pke.unsupervised.YAKE()
            yake.load_document(input=p, language="en", normalization=None)
            yake.candidate_selection(n=1, stoplist=self.stoplist)
            yake.candidate_weighting(window=5, stoplist=self.stoplist, use_stems=False)
            yake_keyphrases = yake.get_n_best(n=self.top_n, threshold=self.threshold)
            candidates.append(yake.candidates)
            topic = self.generate_topic(yake_keyphrases, self.labels[i])
            result.update({self.labels[i]: topic})
        self.candidates = candidates
        return result

    def generate_topic(self, token_rank_dict, label: str):
        topic = Topic(class_name=label)
        for w, v in token_rank_dict:
            topic.add_keyword(keyword=w.split(" "), rank=v, algorithm=self.class_name)
        return topic
