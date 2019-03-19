import numpy as np
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

    def __init__(self, path_to_model: str, embedding_model: EmbeddingModel, label_map = "models/label_maps/3_classes.json"):
        super().__init__(path_to_model)
        self.model = None
        self.embedding_model = embedding_model
        self.label_map = LabelMap(label_map)
        self.initialize()

    def initialize(self) -> None:
        self.model = load(self.path)

    def predict(self, text: str) -> Prediction:
        if isinstance(self.embedding_model, FastTextWrapper):
            embeddings = self.embedding_model.inference(word_tokenize(text), sentence_level=True)
        else:
            embeddings = self.embedding_model.inference(word_tokenize(text))

        probs = self.model.predict_proba(embeddings)
        lm = LabelMap(path_to_json=os.path.join(os.path.dirname(self.path), "label_map.json"))
        classes = []
        values = []

        for x, y in enumerate(probs[0]):
            classes.append(lm.get_name(x))
            values.append(y)

        return Prediction(classes, values)

    def normalize_result(self, prediction: Prediction) -> Prediction:
        pass
