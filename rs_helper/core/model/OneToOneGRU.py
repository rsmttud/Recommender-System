from rs_helper.core.model.Model import Model
from rs_helper.core.Prediction import Prediction
#from rs_helper.classes.FastTextEncoder import FastTextEncoder
from rs_helper.core.Preprocessor import Preprocessor
from rs_helper.core.LabelMap import LabelMap
from keras.models import model_from_yaml
import numpy as np


# Needs to be adjusted to FastTextWrapepr
class OneToOneGRU(Model):

    def __init__(self, path_to_model: str, path_to_weights: str, path_to_encoder: str):
        super().__init__(path_to_model=path_to_model)
        self.model_path = path_to_model
        self.weights_path = path_to_weights
        self.model = None
        self.encoder_path = path_to_encoder

    def initialize(self):
        # model loading
        yaml_file = open(self.model_path, "r")
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        loaded_model.load_weights(self.weights_path)
        self.model = loaded_model

    def predict(self, text: str):
        x = self.__transform_input(text)
        y = self.model.predict(x, verbose=0)
        lm = LabelMap(path_to_json="models/label_maps/3_classes_OneToOne.json")
        preds = [Prediction(values=list(pred[0]), classes=[lm.get_name(i) for i in range(len(pred[0]))]) for pred in y]
        return preds

    def __transform_input(self, text: str):
        pp = Preprocessor(data=text)
        tokens = pp.transform(remove_nums=True, remove_punct=True)
        if len(tokens) > 1:
            e = []
            for s in tokens:
                e.extend(s)
            tokens = [e]
        print(tokens)
        ft = FastTextEncoder(documents=tokens, sentence_level=True)
        vectorized = ft.vectorize()
        x = np.array(vectorized).reshape(shape=(len(vectorized), 1, 100))
        return x

    def normalize_result(self, prediction: Prediction):
        pass
