import tensorflow as tf
from typing import *
from rs_helper.classes.EmbeddingModel import EmbeddingModel, FastText
from rs_helper.classes.Layers import *
from nltk.tokenize import word_tokenize
import os
from sklearn import preprocessing
import pandas as pd


# TODO Check the weight initialisation
# TODO Check the keras Layer Base Class..

class DAN:
    def __init__(self, num_hidden_layers: int,
                 word_embedding_model: EmbeddingModel,
                 model_path: str,
                 wd_prob: float = 0.2,
                 trainable=False):

        self.num_hidden_layers = num_hidden_layers
        self.embedding_model = word_embedding_model
        self.embedding_len = word_embedding_model.inference(["x"]).shape[1]
        self.wd_prob = wd_prob
        self.model_path = model_path

        self.dense_layers = []

        if trainable:
            self.dense_layers = None
            self.train = self.__init_architecture()

    @staticmethod
    def __create_one_hot_encodings(labels):
        le = preprocessing.LabelEncoder()
        labels.apply(le.fit_transform)
        # labels = [le.fit_transform(x) for x in labels]

        enc = preprocessing.OneHotEncoder()
        enc.fit(labels)
        return enc.transform(labels).toarray()

    def __create_dan_layer(self, input_tensor: tf.Tensor):
        dense_layers = []
        for i in range(1, self.num_hidden_layers + 1):
            input = input_tensor
            if len(self.dense_layers) > 0:
                input = self.dense_layers[-1]

            dense_layer = tf.layers.dense(inputs=input,
                                          units=self.embedding_len,
                                          activation=tf.nn.tanh,
                                          use_bias=True,
                                          trainable=True,
                                          name="dense_layer_{}".format(i))
            dense_layers.append(dense_layer)
        return dense_layers

    def __create_classifier_layer(self, input_layer, classifier_shape: List[int]):
        c_layers = []
        for i, neurons in enumerate(classifier_shape):
            if i == 0:
                dense_layer = tf.layers.dense(inputs=input_layer,
                                              units=neurons,
                                              activation=tf.nn.tanh,
                                              use_bias=True,
                                              trainable=True,
                                              name="dense_layer_{}".format(i))
                c_layers.append(dense_layer)
            else:
                dense_layer = tf.layers.dense(inputs=c_layers[-1],
                                              units=neurons,
                                              activation=tf.nn.tanh,
                                              use_bias=True,
                                              trainable=True,
                                              name="dense_layer_{}".format(i))
                c_layers.append(dense_layer)
        return c_layers

    def train(self, text: List[str], labels: List[str], model_name: str, epoches: int = 1, classes: int = 3):

        # Data
        df = pd.DataFrame.from_dict({"text": text, "labels": labels})
        one_hot_encodings = self.__create_one_hot_encodings(df.drop(columns=["text"]))
        # Paths
        save_path = self.model_path
        model_name = model_name

        # Check the Dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Net Architecture
        x_input = tf.placeholder(dtype=tf.float64, shape=(None, 100), name="placeholder_input")
        y_true = tf.placeholder(dtype=tf.float64, shape=(None, classes), name="placeholder_y_true")

        average = average_layer(x_input)
        # Maybe a Function
        dense_dan_layers = self.__create_dan_layer(average)
        classifier_layers = self.__create_classifier_layer(average, [200])

        logits = tf.layers.dense(inputs=classifier_layers[-1],
                                 units=classes,
                                 activation=tf.nn.tanh,
                                 use_bias=True,
                                 trainable=True,
                                 name="logits")

        with tf.name_scope("loss"):
            softmax_layer = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
            cross_entropy = tf.reduce_mean(softmax_layer)
            tf.summary.scalar("cross_entropy", cross_entropy)
        with tf.name_scope("pred_eval"):
            prob_target = tf.argmax(logits, 1, name="prob_target")  # Returns Index with largest value
            prob_true = tf.argmax(y_true, 1, name="prob_true")
            pred_eval = tf.equal(prob_target, prob_true)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            train = optimizer.minimize(cross_entropy, name="train")

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(save_path)
            writer.add_graph(sess.graph)

            preds = []
            for i in range(epoches):
                for j, p in enumerate([word_tokenize(x) for x in text]):
                    # Converting paragraphs to Embeddings
                    embeddings = dropout_layer(self.embedding_model.inference(p))
                    label = np.array(one_hot_encodings[j]).reshape(1, -1)
                    [_, _eval] = sess.run([train, pred_eval], feed_dict={x_input: embeddings, y_true: label})
                    preds.append(_eval[0])
                print("Accuracy: {}".format(np.mean(preds)))
            saver.save(sess, os.path.join(save_path, model_name))

            # Save graph definition
            graph = sess.graph
            graph_def = graph.as_graph_def()
            tf.train.write_graph(graph_def, ".", os.path.join(save_path, model_name + ".pbtxt"), True)

            print("Model saved to {}.".format(os.path.abspath(self.model_path)))

            sess.close()

    def get_dan_embedding(self, text: List[str], layer_name: str) -> np.ndarray:
        """
        Method takes a tokenized sentence as well as the desired output_operation name as parameters
        :param text: Tokenized String
        :param layer_name: Name of last DAN Operation
        :return: np.ndarray (Embedding)
        """
        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(self.model_path + "/first_dan.ckpt.meta")
            # Read tensor from graph
            input = tf.get_default_graph().get_tensor_by_name("placeholder_input:0")
            output = tf.get_default_graph().get_tensor_by_name(layer_name + "/Tanh:0")

            with tf.Session() as sess:
                saver.restore(sess, os.path.join(self.model_path, "first_dan.ckpt"))
                # Get Embeddings from EmbeddingModel
                embeddings = self.embedding_model.inference(text)
                # Run the session
                dan_emb = sess.run(output, feed_dict={input: embeddings})
                sess.close()
                return dan_emb
