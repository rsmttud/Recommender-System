import os
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from tqdm import tqdm
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil


class DatasetGenerator:

    def __init__(self, path_to_files: str, out_path: str,
                 search_terms: list, class_name: str, save_name: str,
                 seperate_save: bool = False, save_num_start: int = 0):
        """
        :param path_to_files: String - Directory where files are stored
        :param out_path: String - Directory where the resulting sentences should be stored
        :param search_terms: List(String) - List of terms that should be searched for in the documents
        :param class_name: Stirng - Class name used for getting topic from pickled topics
        :param save_name: Stirng - Name of the saved files (_<num>.txt gets appended)
        :param seperate_save: Bool - Set if sentences should be stored in one or seperate files
        :param save_num_start: Int - Number from which the saved files numeration should start (See save_name)
        DatasetGenerator takes all files of a directory and searches for sentences that contain specific keywords/
        keyphrases. These will be compared with the topic of the class (Cosine similarity). The top 25% similar
        sentences will be stored in the out_dir.
        """
        if not os.path.exists(path_to_files):
            raise ValueError("No such file or directory")
        self.path = path_to_files
        self.class_name = class_name
        self.result_sentences = list()
        self.search_terms = search_terms
        self.out_path = out_path
        self.save_name = save_name
        self.seperate = seperate_save
        self.save_start = save_num_start

    def run(self):
        sentences = self.find_sentences()
        topics = pickle.load(open("data/manual_topics_for_embedding.pkl", "rb"))
        all_words = self.get_corpora(topics, sentences)
        topic = list(topics[self.class_name])
        topic_vector = self.vectorize(topic, all_words)
        similarities = self.calculate_similarities(all_words, sentences, topic_vector)
        similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:int(len(sentences)*0.25)]
        self.save_in_class(sentences, sim_list=similarities)
        self.save_to_txt(seperate=self.seperate)

    def save_in_class(self, sentences: list, sim_list: list):
        for idx, sim in sim_list:
            self.result_sentences.append(sentences[idx])

    def calculate_similarities(self, basis: list, sentences: list, vector):
        print("Calc Sims:")
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
        print("Getting Copora:")
        all = list()
        for key in topics:
            for w in topics[key]:
                if w not in all:
                    all.append(w)
        for s in sents:
            ws = word_tokenize(s)
            for w in ws:
                if w not in all:
                    all.append(w)
        return list(set(all))

    def find_sentences(self):
        print("Findinf sentences:")
        sents = list()
        for t in tqdm(os.listdir(self.path)):
            if t.endswith(".txt"):
                data = open(os.path.join(self.path, t), "r").read()
                sentences = sent_tokenize(data)
                # print("Sentences: {}".format(len(sentences)))
                for s in sentences:
                    s = re.sub('[^A-Za-z ,.]+', '', s)
                    s.replace("  ", " ")
                    for needle in self.search_terms:
                        if s.find("  ") == -1:
                            if s.lower().find(needle) != -1 and s.lower() not in sents:
                                sents.append(s.lower())
        return sents

    def save_to_dataframe(self):
        df = pd.DataFrame({"sentence": self.result_sentences})
        df["class"] = self.class_name
        return df

    def save_to_txt(self, seperate: bool = False):
        if not seperate:
            file = open(os.path.join(self.out_path, self.save_name + "_sentences.txt"), "w")
            for s in self.result_sentences:
                file.write(s + "\n")
            file.close()
        else:
            if not os.path.exists(os.path.join(self.out_path, self.class_name)):
                os.mkdir(os.path.join(self.out_path, self.class_name))
            for i, s in enumerate(self.result_sentences):
                file = open(os.path.join(self.out_path, self.save_name+"_{}.txt".format(self.save_start+i+1)), "w")
                file.write(s)
                file.close()

    @staticmethod
    def merge_crawl_results(super_dir: str):
        merged_path = os.path.join(super_dir, "merged_datasets")
        if not os.path.exists(merged_path):
            os.mkdir(merged_path)
        for sub_dir in tqdm(os.listdir(super_dir)):
            label_dir = os.path.join(merged_path, sub_dir.split("-")[0])
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
            sub_path = os.path.join(super_dir, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            for file in os.listdir(sub_path):
                if file.endswith("txt"):
                    if file not in os.listdir(label_dir):
                        shutil.copyfile(os.path.join(sub_path, file), os.path.join(label_dir, file))
