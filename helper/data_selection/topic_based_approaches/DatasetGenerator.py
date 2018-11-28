import os
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from tqdm import tqdm
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DatasetGenerator:

    def __init__(self, path_to_files: str, search_term: str, class_name: str, save_name: str):
        if not os.path.exists(path_to_files):
            raise ValueError("No such file or directory")
        self.path = path_to_files
        self.class_name = class_name
        self.result_sentences = list()
        self.search_term = search_term
        self.save_name = save_name

    def run(self):
        sentences = self.find_sentences()
        topics = pickle.load(open("manual_topics_for_embedding.pkl", "rb"))
        all_words = self.get_corpora(topics, sentences)
        clustering_topic = list(topics[self.class_name])
        clustering_vector = self.vectorize(clustering_topic, all_words)
        similarities = self.calculate_similarities(all_words, sentences, clustering_vector)
        similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:int(len(sentences)*0.25)]
        self.save_in_class(sentences, sim_list=similarities)
        self.save_to_txt()

    def save_in_class(self, sentences: list, sim_list: list):
        for idx, sim in sim_list:
            self.result_sentences.append(sentences[idx])

    def calculate_similarities(self, basis: list, sentences: list, vector):
        sims = dict()
        for i, s in enumerate(tqdm(sentences)):
            vec = self.vectorize(word_tokenize(s), basis)
            similarity = cosine_similarity(vec.reshape(1, -1), vector.reshape(1, -1))
            sims.update({i: similarity[0][0]})
        return sims

    def vectorize(self, words, basis):
        def increase(vec, index):
            vec[index] += 1
        vector = np.zeros(len(basis))
        [increase(vector, i) for i, w in enumerate(basis) if w in words]
        return vector

    def get_corpora(self, topics, sents):
        all = list()
        for key in topics:
            for w in topics[key]:
                if w not in all:
                    all.append(w)
        print("Getting words of sentences:")
        for s in tqdm(sents):
            ws = word_tokenize(s)
            for w in ws:
                if w not in all:
                    all.append(w)
        return list(set(all))

    def find_sentences(self):
        sents = list()
        for t in tqdm(os.listdir(self.path)):
            if t.endswith(".txt"):
                data = open(os.path.join(self.path, t), "r").read()
                sentences = sent_tokenize(data)
                for s in sentences:
                    s = re.sub('[^A-Za-z ]+', '', s)
                    if s.lower().find(self.search_term) != -1:
                        sents.append(s.lower())
        return sents

    def save_to_dataframe(self):
        df = pd.DataFrame({"sentence": self.result_sentences})
        df["class"] = self.class_name
        return df

    def save_to_txt(self):
        file = open(self.save_name + "_sentences.txt", "w")
        for s in self.result_sentences:
            file.write(s + "\n")
        file.close()


if __name__ == "__main__":
    generator = DatasetGenerator(path_to_files="../../data_obtaining/science_direct/out/sequence analysis",
                                 search_term="sequence analysis is",
                                 class_name="sequence_analysis",
                                 save_name="sequence_analysis")
    generator.run()

