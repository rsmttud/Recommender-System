import pickle
import os
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from rs_helper.classes import Corpora
from nltk.tokenize import word_tokenize
from functools import reduce
import operator

TOPIC_DIR = "data/topics/"


def contains_list(l):
    for el in l:
        if not isinstance(el, str):
            print(el)


topic_dict = dict()
for t in [x for x in os.listdir(TOPIC_DIR) if x.endswith(".topic")]:
    topic = pickle.load(open(os.path.join(TOPIC_DIR, t), "rb"))
    keyword_names = [" ".join(x) for x in topic.get_keyword_names() if isinstance(x, list)]
    keyword_names_2 = [x for x in topic.get_keyword_names() if isinstance(x, str)]
    keyword_names_full = keyword_names + keyword_names_2
    topic_dict[t.split(".")[0]] = keyword_names_full

t_clustering = topic_dict["clustering"]
t_prediction = topic_dict["prediction"]
t_pattern_mining = topic_dict["pattern_mining"]

tfidf_vocab = pickle.load(open("data/topics/tfidf_vocab.vocab", "rb"))
tfigm_vocab = pickle.load(open("data/topics/tfigm_vocab.vocab", "rb"))
topicRank_vocab = pickle.load(open("data/topics/topicRank_vocab.vocab", "rb"))
topicRank_vocab = reduce(operator.concat, [list(x.keys()) for x in topicRank_vocab])
vocab = reduce(operator.concat, [tfigm_vocab, tfidf_vocab, topicRank_vocab])

dictionary = Dictionary(vocab)
corpus = dictionary.doc2bow(vocab)

contains_list(t_clustering)
contains_list(t_pattern_mining)
contains_list(t_prediction)

print(dictionary.token2id["clustering"])
cm = CoherenceModel(topics=[t_clustering, t_pattern_mining, t_prediction], corpus=corpus, dictionary=dictionary, coherence='u_mass')
coherence = cm.get_coherence()
print(coherence)
