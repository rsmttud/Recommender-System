import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rs_helper.core import *
from rs_helper.helper import *
from rs_helper.helper.classes.EmbeddingModel import EmbeddingModel, FastTextWrapper, DAN
from rs_helper.helper.functions.visualizations import *


class LookupTable:
    def __init__(self, tokens: list, model):
        self._len = len(tokens)
        self._tokens = tokens
        self._model = model
        self.table = dict()
        self.initialize()

    def initialize(self):
        embeddings = self._model.inference(self._tokens)
        # print(embeddings)
        for t, v in zip(self._tokens, embeddings):
            self.table[t] = v

    def lookup(self, t: str):
        if len(self.table) == 0:
            raise ValueError("Lookup Table needs to be initialized first.")
        return self.table[t]


class SimilarityEval:
    def __init__(self, path, sim_data, valid_data, **kwargs):
        """
        :param path: path to FastText Model
        :param sim_data: path to Similarity CSV
        :param valid_data: path to validation data
        Class to perform similarity evaluation
        """
        self.path = path
        self.base_path = os.path.dirname(path)
        self.folder = os.path.basename(os.path.dirname(path))

        if path.endswith(".pb"):
            self.ft_path = kwargs.get("ft_path", "")
            self.ft_model = FastTextWrapper(path=self.ft_path)
            self.model = DAN(word_embedding_model=self.ft_model, frozen_graph_path=path)
            self.dan = True
        elif path.endswith(".joblib"):
            self.model = FastTextWrapper(path=path)
            self.dan = False
        else:
            raise ValueError("Please specify correct model path")

        self.sim_data = sim_data
        self.valid_data = valid_data
        self.pearson_corr = None
        self.spearman_corr = None
        self.ME = None

    def calculate_similarities(self):
        """
        :return: List(float): List of similarity values
        Calculate similarities for word pairs in similiarity data
        """
        sims = list()
        side_1 = list(self.sim_data["Word 1"])
        side_2 = list(self.sim_data["Word 2"])

        lt = LookupTable(tokens=side_1+side_2, model=self.model)

        for i, (w1, w2) in enumerate(zip(side_1, side_2)):
            v1 = lt.lookup(w1)
            v2 = lt.lookup(w2)

            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                sims.append(cosine_similarity([v1], [v2])[0][0])
            else:
                sims.append(np.nan)

        self.sim_data["assigned_sim"] = sims
        self.sim_data = self.sim_data.dropna()
        self.mean_error()
        self.correlation()
        return sims

    def mean_error(self):
        """
        :return: float
        Calculate mean error of similarity judgements
        """
        self.ME = sum(np.array(self.sim_data["Human (mean)"]) - np.array(self.sim_data["assigned_sim"])) / len(
            self.sim_data)
        return self.ME

    def correlation(self):
        """
        :return: List(float)
        Correlation between human similarities and model similarities
        """
        self.pearson_corr = self.sim_data["Human (mean)"].corr(self.sim_data["assigned_sim"], method="pearson")
        self.spearman_corr = self.sim_data["Human (mean)"].corr(self.sim_data["assigned_sim"], method="spearman")
        return [self.pearson_corr, self.spearman_corr]

    def save_to_config(self):
        """
        :return: None
        Save similarity judgement values in config file
        """
        config_path = os.path.join(self.base_path, "config.json")

        with open(config_path, "r") as _json:
            c_dict = json.load(_json)

        c_dict["mean_similarity_error"] = self.ME
        c_dict["similarity_correlation"] = self.pearson_corr
        c_dict["similarity_spearman_correlation"] = self.spearman_corr

        with open(config_path, "w") as _json:
            json.dump(c_dict, _json)

    def plot_similarity(self):
        """
        :return: None
        Create and save similarity matrix of valid data
        """
        if self.dan:
            self.valid_data["vector"] = self.valid_data["text"].apply(
                lambda x: self.model.inference(x)[0])
        else:
            self.valid_data["vector"] = self.valid_data["text"].apply(
                lambda x: self.model.inference(word_tokenize(x), sentence_level=True))

        messages = list(self.valid_data["label"])
        vectors = list(self.valid_data["vector"])
        similarity_matrix(messages=messages, vectors=vectors, name=self.folder, save_path=self.base_path)
