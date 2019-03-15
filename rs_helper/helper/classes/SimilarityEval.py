import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rs_helper.core import *
from rs_helper.helper import *


class SimilarityEval:
    def __init__(self, path: str, sim_data: pd.DataFrame, valid_data: pd.DataFrame) -> None:
        """
        :param path: path to FastText model
        :param sim_data: path to Similarity CSV
        :param valid_data: path to validation data
        Class to perform similarity evaluation
        """
        self.path = path
        self.base_path = os.path.dirname(path)
        self.folder = os.path.basename(os.path.dirname(path))
        self.model = FastTextWrapper(path=path)
        self.sim_data = sim_data
        self.valid_data = valid_data
        self.pearson_corr = None
        self.spearman_corr = None
        self.ME = None

    def calculate_similarities(self) -> List[float]:
        """
        :return: List(float): List of similarity values
        Calculate similarities for word pairs in similiarity data
        """
        sims = list()
        for i, r in self.sim_data.iterrows():
            vecs = self.model.inference([r["Word 1"], r["Word 2"]])
            if len(vecs) == 2:
                sims.append(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
            else:
                sims.append(np.nan)
        self.sim_data["assigned_sim"] = sims
        self.sim_data = self.sim_data.dropna()
        self.mean_error()
        self.correlation()
        return sims

    def mean_error(self) -> float:
        """
        :return: float
        Calculate mean error of similarity judgements
        """
        self.ME = sum(np.array(self.sim_data["Human (mean)"]) - np.array(self.sim_data["assigned_sim"])) / len(
            self.sim_data)
        return self.ME

    def correlation(self) -> List[float, float]:
        """
        :return: List(float)
        Correlation between human similarities and model similarities
        """
        self.pearson_corr = self.sim_data["Human (mean)"].corr(self.sim_data["assigned_sim"], method="pearson")
        self.spearman_corr = self.sim_data["Human (mean)"].corr(self.sim_data["assigned_sim"], method="spearman")
        return [self.pearson_corr, self.spearman_corr]

    def save_to_config(self) -> None:
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

    def plot_similarity(self) -> None:
        """
        :return: None
        Create and save similarity matrix of valid data
        """
        self.valid_data["vector"] = self.valid_data["text"].apply(
            lambda x: self.model.inference(word_tokenize(x), sentence_level=True))
        messages = list(self.valid_data["label"])
        vectors = list(self.valid_data["vector"])
        similarity_matrix(messages=messages, vectors=vectors, name=self.folder, save_path=self.base_path)
