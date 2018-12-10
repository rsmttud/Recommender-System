from rs_helper.classes.Prediction import Prediction
from rs_helper.classes.Model import Model
import tensorflow as tf
import tensorflow_hub as tf_hub
import pandas as pd
import os


class EmbeddingClassificationPipeline(Model):

    def __init__(self, path_to_model: str, path_to_embedding_model: str, num_classes=4):
        super().__init__(path_to_model)
        self.path_to_classifier = path_to_model
        self.path_to_embedding_model = path_to_embedding_model
        self.num_classes = num_classes

        self.estimator = None

    def initialize(self) -> None:
        # TODO
        os.environ["TFHUB_CACHE_DIR"] = '/tfhub'
        tf.logging.set_verbosity(tf.logging.ERROR)

        model_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        #model = tf_hub.Module(model_url, trainable=False)

        embedded_text_feature_column = tf_hub.text_embedding_column(
            key="sentence",
            module_spec=model_url,
            trainable=True)

        self.estimator = tf.estimator.DNNClassifier(
            hidden_units=[500, 100],
            feature_columns=[embedded_text_feature_column],
            warm_start_from=self.path_to_classifier,
            n_classes=4,
            optimizer=tf.train.AdagradOptimizer(learning_rate=0.001))

    def normalize_result(self, prediction: Prediction) -> Prediction:
        pass

    def __text_to_input_func(self, text):
        df = pd.DataFrame.from_dict({"sentence": [text]})
        pred_input_func = tf.estimator.inputs.pandas_input_fn(df, shuffle=False)
        return pred_input_func

    def predict(self, text: str) -> Prediction:
        pred_input_func = self.__text_to_input_func(text)
        result = self.estimator.predict(pred_input_func)
        classes = list(range(0, self.num_classes))

        return Prediction(classes, values=list(list(result)[0]["probabilities"]))
