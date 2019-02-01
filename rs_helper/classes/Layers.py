import tensorflow as tf
import numpy as np


def dropout_layer(input_layer: np.ndarray, dropout_prob: float = 0.2):
    # dist = tf.contrib.distributions.Binomial(1, self.drop)
    dist = 1 - np.random.binomial(1, dropout_prob, input_layer.shape[0])
    return input_layer[dist.astype(bool)]


def average_layer(placeholder: tf.Variable, name="average_layer"):
    with tf.name_scope(name):
        average = tf.reduce_mean(input_tensor=placeholder, axis=0)
        return tf.reshape(average, (1, -1), name="average_layer")
