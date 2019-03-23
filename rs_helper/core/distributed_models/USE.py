import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
from typing import List, Any
from rs_helper.core.distributed_models.EmbeddingModel import EmbeddingModel


class USE(EmbeddingModel):

    def __init__(self, **kwargs):
        """
        Universal Sentence Encoder Object

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.tf_model = self.initialize_model()

    def initialize_model(self, **kwargs) -> tf_hub.Module:
        """
        Initializes the model by downloading it if not already available.

        :param kwargs:

        :return: tf_hub.Module
        :rtype: tf_hub.Module
        """
        url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        return tf_hub.Module(url, trainable=False)

    def inference(self, words: List[str]) -> np.ndarray:
        """
        Returns embeddings given a list of strings. You can also inference batches with this method..

        :param words: In this case it  can be something like this: ["hello", "Im a paragraph"]
        :type words: list(str)

        :return: the inferred embeddings
        :rtype: np.ndarray
        """
        session_conf = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        with tf.Session(config=session_conf) as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            with tf.device('/gpu:0'):
                message_embeddings = session.run(self.tf_model(words))

            return message_embeddings
