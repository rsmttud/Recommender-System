import tensorflow as tf
from typing import *
from rs_helper.classes.EmbeddingModel import EmbeddingModel, FastText
from rs_helper.classes.Layers import WordDropoutLayer, AverageLayer


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

    def train(self, text: List[List[str]], text_labels: Any, classifier_hidden_units:List[int]):
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
            input_layer = self.embedding_model.inference(paragraph)  # Getting the Embeddings
            dropout_layer = WordDropoutLayer(input_layer,
                                             0.2).drop_word()  # TODO Check the keras Layer Base Class..
            averaging_layer = AverageLayer(dropout_layer).average_layer()  # Averaging Embeddings Placeholder..
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            with tf.Session() as sess:
                sess.run(init)
                u = sess.run(averaging_layer)  # Remove the variable..
                hidden_states = sess.run(self.dense_layers[-1], feed_dict={X: u})
                print(hidden_states)
