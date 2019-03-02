import os
from gensim.models import FastText
import pickle
from rs_helper.classes import FastTextEncoder
from rs_helper.classes import KeywordExtractionPipeline

sent = "I want to group my customers into different segments. This should be done to " \
       "receive different categories of customers."

kwp = KeywordExtractionPipeline(path_to_topics="models/topics/")
kwp.initialize()
print(kwp.topic_keywords)
print(kwp.predict(sent))
