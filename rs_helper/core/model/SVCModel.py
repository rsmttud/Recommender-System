import numpy as np
from sklearn.svm import SVC
import os
from rs_helper.core.model.Model import Model
from rs_helper.core.Prediction import Prediction
from rs_helper.core.distributed_models.EmbeddingModel import EmbeddingModel
from rs_helper.core.distributed_models.DAN import DAN
from rs_helper.core.LabelMap import LabelMap
from rs_helper.core.distributed_models.FastTextWrapper import FastTextWrapper
from joblib import load
from nltk.tokenize import word_tokenize
from typing import List


class SVCModel(Model):

    def __init__(self, path_to_model: str,
                 embedding_model: EmbeddingModel):

        super().__init__(path_to_model)
        self.model = None
        self.embedding_model = embedding_model
        self.label_map = None
        self.initialize()

    def initialize(self) -> None:
        self.model = load(self.path)
        self.label_map = LabelMap(path_to_json=os.path.join(os.path.dirname(self.path), "label_map.json"))
        if not isinstance(self.model, SVC):
            raise ValueError("Supplied model not of type sklearn.svm.SVC")

    def predict(self, text: str) -> Prediction:
        if isinstance(self.embedding_model, FastTextWrapper):
            embeddings = self.embedding_model.inference(word_tokenize(text), sentence_level=True)
        else:
            embeddings = self.embedding_model.inference(word_tokenize(text))

        probs = self.model.predict_proba(embeddings)
        classes = []
        values = []

        for x, y in enumerate(probs[0]):
            classes.append(self.label_map.get_name(x))
            values.append(y)

        return Prediction(classes, values)
