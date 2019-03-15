from rs_helper.core.model.Model import Model
from rs_helper.core.Prediction import Prediction
from rs_helper.core.distributed_models import EmbeddingModel, DAN, FastTextWrapper
from nltk.tokenize import word_tokenize
from rs_helper.core.LabelMap import LabelMap
from keras.models import model_from_yaml
import numpy as np
import os


class RNNTypedClassifier(Model):

    def __init__(self, path_to_model: str, architecture: str, embedding_type: str):
        """
        General class for RNNbased Architectures. Supported are One-to-One GRU / LSTMs and Many-to-One LSTM / GRUs

        :param path_to_model: Path to the model directory. Needs to contain model.yaml, weights.h5 and label_map.json
        :type path_to_model: str
        :param architecture: "N:1" or "1:1" Architecture
        :type architecture: str
        :param embedding_type: "DAN" or "FastText" Embeddings
        :type embedding_type: str

        """
        if not os.path.isdir(path_to_model):
            raise NotADirectoryError("The specified path is not a directory")

        paths = [os.path.join(path_to_model, x) for x in ["model.yaml", "weights.h5", "label_map.json"]]
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError("The supplied directory does not contain {}. The directory needs to contain "
                                        "the files model.yaml, weights.h5 and label_map.json "
                                        "at least.".format(os.path.basename(p)))

        if architecture not in ["N:1", "1:1"]:
            raise ValueError("Specified architecture not available. Please select N:1 or 1:1.")

        if embedding_type not in ["DAN", "FastText"]:
            raise ValueError("Specified embedding_type not available. Please select DAN or FastText")

        super().__init__(path_to_model=path_to_model)
        self.model_path = path_to_model
        self.model = None
        self.architecture = architecture
        self.embedding_type = embedding_type

    def initialize(self):
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
        pred = Prediction(values=list(y), classes=[lm.get_name(i) for i in range(len(y))])
        return pred

    def __transform_input(self, text: str):
        """
        Transform input to the shape (len(text), 1, 100) or (len(text), 100) depending on architecture.

        :param text: The text to transform
        :type text: str

        :return: The input for classification
        :rtype: np.ndarray
        """
        tokens = word_tokenize(text)
        vectorized = self.__get_vetorized_text(tokens)
        if self.architecture == "1:1":
            x = np.array(vectorized).reshape(shape=(len(vectorized), 1, 100))
        else:
            x = np.array(vectorized).reshape(shape=(len(vectorized), 100))
        return x

    def __get_vetorized_text(self, tokens: list):
        """
        Uses Embedding models to receive the vector representations

        :param tokens: the tokens to transform in vectors
        :type tokens: list(str)

        :return: list of embeddings
        :rtype: list(np.ndarray)
        """
        _FT = FastTextWrapper(path="models/FastText/1/model.joblib")
        _DAN = None
        if self.embedding_type == "DAN":
            _DAN = DAN(frozen_graph_path="models/DANs/1/frozen_graph.pb", word_embedding_model=_FT)

        if self.architecture == "1:1":
            vectorized = _FT.inference(words=tokens, sentence_level=True) \
                if self.embedding_type == "FastText" else _DAN.inference_batches([tokens])
        else:
            vectorized = [_FT.inference(x) for x in tokens] \
                if self.embedding_type == "FastText" else [_DAN.inference(x) for x in tokens]
        return vectorized

    def normalize_result(self, prediction: Prediction):
        pass
