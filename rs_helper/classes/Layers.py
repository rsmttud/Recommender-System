import random
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


class Batches:

    def __init__(self, x, y, batch_size=8):
        self.__x = x
        self.__y = y
        self.__data = self.__init_data()
        self.__batch_size = batch_size
        self.__current_state = 0

    def __init_data(self):
        lst = list(zip(self.__x, self.__y))
        arr = np.array(lst)
        np.random.shuffle(arr)
        return arr

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_state >= self.__data.shape[0]:
            raise StopIteration
        else:
            batch = self.__data[self.__current_state: self.__current_state + self.__batch_size]
            x = batch[:, 0]
            y = batch[:, 1]
            self.__current_state += self.__batch_size
            return x, y
