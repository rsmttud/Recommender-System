import tensorflow as tf
# from keras.layers import Layer
import numpy as np
from nltk.tokenize import word_tokenize
from typing import *
import pickle
import random
from abc import ABC, abstractmethod
from nltk.tokenize import sent_tokenize

# TODO Batch Size
# Parameters for DAN
"""
- Hidden Layer. 
- Embedding Shape? I dont need that i get the Information from the incoming Embeddings
- Activation Function
- Dropout?
- Batch Normalization? Not in the papers... 
- Word Dropout probability. 
- batch_size
- optimizer
"""


class EmbeddingModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __initialize_model(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, words: List[str]) -> np.ndarray:
        pass


class DAN:
    def __init__(self, num_hidden_layers: int,
                 word_embedding_model: EmbeddingModel,
                 activation_func: Any = tf.nn.tanh,
                 wd_prob: float = 0.2):

        self.num_hidden_layers = num_hidden_layers
        self.activation_func = activation_func
        self.embedding_model = word_embedding_model
        self.embedding_len = 100  # TODO HARDCODED
        self.wd_prob = wd_prob
        self.dense_layers = []

    def train(self, text: List[List[str]]):
        X = tf.placeholder(dtype=tf.float64, shape=(None, 100))  # Input which is feed into the dense networks.
        # Dense Layers
        # TODO Check the weight initialisation
        for i in range(1, self.num_hidden_layers + 1):
            input = X
            if len(self.dense_layers) > 0:
                input = self.dense_layers[-1]

            dense_layer = tf.layers.dense(inputs=input,
                                          units=self.embedding_len,
                                          activation=self.activation_func,
                                          use_bias=True,
                                          trainable=True,
                                          name="dense_layer_{}".format(i))
            self.dense_layers.append(dense_layer)



        ### TODO  MAKE IT PRETTY CLOSE THE SESSION!
        for paragraph in text:
            input_layer = fast_text_model.inference(paragraph)  # Getting the Embeddings
            dropout_layer = WordDropoutLayer(input_layer,
                                             0.2).drop_word()  # TODO Check the keras Layer Base Class..
            averaging_layer = AverageLayer(dropout_layer).average_layer()  # Averaging Embeddings
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            with tf.Session() as sess:
                sess.run(init)
                u = sess.run(averaging_layer) # Remove the variable..
                hidden_states = sess.run(self.dense_layers[-1], feed_dict={X: u})
                print(hidden_states)


class AverageLayer:
    """
    The layer which averages the  word Embeddings.
    """

    def __init__(self, embeddings: tf.Variable):
        self.embeddings = embeddings
        # TODO Check if it averages the right way!
        self.average = tf.reduce_mean(input_tensor=self.embeddings, axis=0)

    def average_layer(self) -> tf.Variable:
        return tf.Variable(tf.reshape(self.average, (1,-1)))


class WordDropoutLayer:
    """
    Randomly drop some words of the Embeddings LST
    """

    def __init__(self, embeddings: np.ndarray, probability: float = 0.2):
        self.embeddings = embeddings
        self.dropout_prob = probability

    def drop_word(self) -> tf.Variable:
        mask = 1 - np.random.binomial(1, self.dropout_prob, self.embeddings.shape[0])
        return tf.Variable(self.embeddings[mask.astype(bool)])


class FastText():
    """
    Loads the FastText model and get the Vectors.
    """

    def __init__(self, **kwargs):
        #super().__init__(**kwargs)
        self.path = "/Users/Daniel/PycharmProjects/Recommender-System/notebooks/FastText/ft_model_15000.pkl"
        self.__initialize_model()

    def __initialize_model(self, **kwargs):
        try:
            tf.logging.info("FastText Model is loading")
            self.model = pickle.load(open(self.path, "rb"))
            tf.logging.info("FastText Model loaded!")
        except Exception as e:
            tf.logging.warning("Something went wrong while loading the FastText Model..")
            tf.logging.warning(e)

    def inference(self, words: List[str]) -> np.ndarray:
        embeddings = []
        for word in words:
            if self.model.wv.__contains__(word):
                embeddings.append(self.model.wv.__getitem__(word))
        return np.array(embeddings)


if __name__ == '__main__':
    ft_path = "/Users/Daniel/PycharmProjects/Recommender-System/notebooks/FastText/ft_model_15000.pkl"
    fast_text_model = FastText()
    words = [["this", "sucks"], ["hello", "my", "name", "is", "daniel"]]

    dan = DAN(num_hidden_layers=2, word_embedding_model=fast_text_model)
    dan.train(words)

    """
    ft_path = "/Users/Daniel/PycharmProjects/Recommender-System/notebooks/FastText/ft_model_15000.pkl"
    fast_text_model = FastText(ft_path)

    words = ["hello", "my", "name", "is", "daniel"]
    embeddings = fast_text_model.inference(words)
    drop_l = WordDropoutLayer(embeddings, 0.2).drop_word()
    average_l = AverageLayer(drop_l).average

    placeholder = tf.placeholder(dtype=tf.float64, shape=(None, 100))
    dense_1 = tf.layers.dense(inputs=placeholder,
                              units=100,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              trainable=True)

    dense_2 = tf.layers.dense(inputs=dense_1,
                              units=100,
                              activation=tf.nn.tanh,
                              use_bias=True,
                              trainable=True)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        x = sess.run(average_l)

        result = sess.run(dense_2, feed_dict={placeholder: x.reshape(1, -1)})

        print(result)
        # print(result.shape)
        # print(result)
        # print(sess.run(mean))
        
    """
