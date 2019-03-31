from rs_helper.core.model.Model import Model
from rs_helper.core.Prediction import Prediction
from rs_helper.core.distributed_models import EmbeddingModel, DAN, FastTextWrapper
from nltk.tokenize import word_tokenize
from rs_helper.core.LabelMap import LabelMap
from keras.models import model_from_yaml
import numpy as np
from typing import List, Any
import os


class SequencePadder:

    def __init__(self, distance: int) -> None:
        """
        Class that can be used to pad sequences to specific lengths.

        :param distance: The length sequences should be padded to
        :type distance: int
        """
        self.dist = distance

    def pad(self, sequence: Any) -> np.ndarray:
        """
        Pads a given sequence to length self.distance

        :param sequence: The sequence to pad
        :type sequence: Any

        :return: Padded sequence
        :rtype: np.ndarray
        """

        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)

        if len(sequence) < self.dist:
            to_pad = self.dist - len(sequence)
            padding = np.array([np.zeros(100) for _ in range(to_pad)])
            seq = np.concatenate((sequence, padding))
        else:
            seq = sequence[:self.dist]
        return seq


class RNNTypedClassifier(Model):

    def __init__(self, model_dir: str, architecture: str, embedding_model: EmbeddingModel) -> None:
        """
        General class for RNNbased Architectures. Supported are One-to-One GRU / LSTMs and Many-to-One LSTM / GRUs

        :param model_dir: Path to the model directory. Needs to contain model.yaml, weights.h5 and label_map.json
        :type model_dir: str
        :param architecture: "N:1" or "1:1" Architecture
        :type architecture: str
        :param embedding_model: Initialized EmbeddingModel
        :type embedding_model: EmbeddingModel

        """
        if not os.path.isdir(model_dir):
            raise NotADirectoryError("The specified path is not a directory")

        paths = [os.path.join(model_dir, x) for x in ["model.yaml", "weights.h5", "label_map.json"]]
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError("The supplied directory does not contain {}. The directory needs to contain "
                                        "the files model.yaml, weights.h5 and label_map.json "
                                        "at least.".format(os.path.basename(p)))

        if architecture not in ["N:1", "1:1"]:
            raise ValueError("Specified architecture not available. Please select N:1 or 1:1.")

        super().__init__(path_to_model=model_dir)
        self.model_path = model_dir
        self.model = None
        self.architecture = architecture
        self.embedding_model = embedding_model
        self.padder = SequencePadder(distance=69)
        self.initialize()

    def initialize(self) -> None:
        # model loading
        yaml_file = open(os.path.join(self.model_path, "model.yaml"), "r")
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        loaded_model.load_weights(os.path.join(self.model_path, "weights.h5"))
        self.model = loaded_model

    def predict(self, text: str) -> Prediction:
        x = self.__transform_input(text)
        y = self.model.predict(x, verbose=0)
        lm = LabelMap(path_to_json=os.path.join(self.model_path, "label_map.json"))
        pred = Prediction(values=list(y[0]), classes=[lm.get_name(i) for i in range(len(y[0]))])
        return pred

    def __transform_input(self, text: str) -> np.ndarray:
        """
        Transform input to the shape (len(text), 1, 100) or (len(text), 100) depending on architecture.

        :param text: The text to transform
        :type text: str

        :return: The input for classification
        :rtype: np.ndarray
        """

        if self.architecture == "1:1":
            vectorized = self.__get_vetorized_text([text])
            x = np.array(vectorized).reshape((1, 1, 100))
        else:
            tokens = word_tokenize(text)
            vectorized = self.__get_vetorized_text(tokens)
            vectorized = self.padder.pad(vectorized)
            x = np.array(vectorized).reshape((1, self.padder.dist, 100))
        return x

    def __get_vetorized_text(self, tokens: List[str]) -> np.ndarray:
        """
        Uses Embedding models to receive the vector representations

        :param tokens: the tokens to transform in vectors
        :type tokens: list(str)

        :return: list of embeddings
        :rtype: list(np.ndarray)
        """
        return self.embedding_model.inference(tokens)

