from rs_helper.core.Model.Model import Model
from rs_helper.core.Prediction import Prediction
from rs_helper.core.LabelMap import LabelMap
import pickle


# TODO needs an adjustment for Embeddings..
class SVC(Model):

    def __init__(self, path_to_model: str, path_to_vectorizer: str):
        super().__init__(path_to_model=path_to_model)
        self.vect = pickle.load(open(path_to_vectorizer, "rb"))

    def initialize(self):
        self.model = pickle.load(open(self.path, "rb"))

    # TODO Talk about how to implement preprocessing
    # TODO -> Using the TFIDFVectorizer of Sklearn with included preprocessing
    def predict(self, text: str):
        input = self.vect.transform([text])
        resulting_class = self.model.predict(input)[0]
        lm = LabelMap(path_to_json="models/label_maps/3_classes.json")
        id = lm.get_index(resulting_class)
        v = [0, 0, 0]
        v[id] = 1
        pred = Prediction(classes=[0, 1, 2], values=v)
        return pred

    def normalize_result(self, prediction: Prediction):
        pass
