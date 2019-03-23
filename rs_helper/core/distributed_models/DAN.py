import os
from typing import *
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from rs_helper.core.distributed_models.EmbeddingModel import EmbeddingModel
from rs_helper.helper.classes.tf_helper import Batches, dropout_layer


# OPTIMIZED FOR GPU TRAINING!
class DAN(EmbeddingModel):
    def __init__(self,
                 word_embedding_model: EmbeddingModel,
                 frozen_graph_path: str = "",
                 **kwargs) -> None:
        """
        Object for training and using Deep Averaging Networks.

        :param word_embedding_model: The embedding model that is used to generate input vectors
        :type word_embedding_model: EmbeddingModel
        :param frozen_graph_path: Path to the frozen_graph.pb of the DAN model
        :type frozen_graph_path: str
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.embedding_model = word_embedding_model
        self.embedding_len = word_embedding_model.inference(["x"]).shape[1]
        self.frozen_graph_path = frozen_graph_path
        self.config = {}

    @staticmethod
    def create_one_hot_encodings(labels: pd.DataFrame) -> np.ndarray:
        """
        Method to create One-Hot-representations of labels

        :param labels: The labels to encode
        :type labels: list(str))

        :return: The encoded labels
        :rtype: np.ndarray
        """
        le = preprocessing.LabelEncoder()
        labels.apply(le.fit_transform)
        enc = preprocessing.OneHotEncoder()
        enc.fit(labels)
        return enc.transform(labels).toarray()

    def initialize_model(self, **kwargs) -> tf.Graph:
        """
        Method to initialize the DAN model

        :param kwargs:

        :return: the tensorflow graph
        :rtype: tf.Graph
        """
        # Parse Protobuf
        frozen_graph_path = self.frozen_graph_path
        with tf.gfile.GFile(frozen_graph_path, "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())

        # import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def __create_training_pipe(self, classes: int, classifier_shape: List[int], num_hidden_layer: int,
                               classifier_act: Any):
        # Updating config dict
        self.config.update(
            {"classes": str(classes), "classifier_shape": str(classifier_shape), "hidden_layer": str(num_hidden_layer)})
        # Net Architecture
        x_input = tf.placeholder(dtype=tf.float64, shape=(None, self.embedding_len), name="placeholder_input")
        y_true = tf.placeholder(dtype=tf.float64, shape=(None, classes), name="placeholder_y_true")

        # Composition Layer
        # average = average_layer(x_input)

        # Dense Layers
        dense_dan_layers = self.__create_dan_layer(x_input, num_hidden_layer=num_hidden_layer)
        # self.last_dan_layer = dense_dan_layers[-1].name

        # Classifier
        # TODO Activation Function
        classifier_layers = self.create_classifier_layer(dense_dan_layers[-1], classifier_shape, classifier_act)

        logits = tf.layers.dense(inputs=classifier_layers[-1],
                                 units=classes,
                                 activation=classifier_act,
                                 use_bias=True,
                                 trainable=True,
                                 name="logits")

        with tf.name_scope("loss"):
            softmax_layer = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
            cross_entropy = tf.reduce_mean(softmax_layer)
            tf.summary.scalar("cross_entropy", cross_entropy)
        """    
        with tf.name_scope("pred_eval"):
            prob_target = tf.argmax(logits, 1, name="prob_target")  # Returns Index with largest value
            prob_true = tf.argmax(y_true, 1, name="prob_true")
            pred_eval = tf.equal(prob_target, prob_true)
        """

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            train = optimizer.minimize(cross_entropy, name="train")

        return train, x_input, y_true, dense_dan_layers[-1]

    def __create_dan_layer(self, input_tensor: tf.Tensor, num_hidden_layer) -> List[tf.layers.dense]:
        dense_layers = []
        dense_layers.append(input_tensor)
        for i in range(1, num_hidden_layer + 1):
            dense_layer = tf.layers.dense(units=self.embedding_len,
                                          inputs=dense_layers[-1],
                                          activation=tf.nn.tanh,
                                          use_bias=True,
                                          trainable=True,
                                          name="dense_layer_{}".format(i))
            dense_layers.append(dense_layer)
        return dense_layers

    @staticmethod
    def create_classifier_layer(input_layer, classifier_shape: List[int], classifier_act) -> List[tf.layers.dense]:
        c_layers = []
        for i, neurons in enumerate(classifier_shape):
            if i == 0:
                dense_layer = tf.layers.dense(inputs=input_layer,
                                              units=neurons,
                                              activation=classifier_act,
                                              use_bias=True,
                                              trainable=True,
                                              name="cl_layer_{}".format(i))
                c_layers.append(dense_layer)
            else:
                dense_layer = tf.layers.dense(inputs=c_layers[-1],
                                              units=neurons,
                                              activation=classifier_act,
                                              use_bias=True,
                                              trainable=True,
                                              name="cl_layer_{}".format(i))
                c_layers.append(dense_layer)
        return c_layers

    def train(self,
              text: List[str],
              labels: List[str],
              classifier_shape: List[int],
              save_path: str,
              model_name: str,
              epoches: int = 1,
              classes: int = 3,
              num_hidden_layer: int = 3,
              classifier_act=tf.nn.tanh,
              batch_size: int = 1,
              wdrop_prob: float = 0.2
              ) -> None:

        self.config.update({"epoches": str(epoches),
                            "dropout": str(wdrop_prob),
                            "path": os.path.abspath(os.path.join(save_path, model_name))})

        # TODO don't use a DataFrame here
        # Bundle the Data together in a DataFrame
        df = pd.DataFrame.from_dict({"text": text, "labels": labels})
        # Create One-Hot-Encodings for Available Classes
        one_hot_encodings = self.create_one_hot_encodings(df.drop(columns=["text"]))

        # Check the if dir exist
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Get Training Pipe and Placeholder
        train_pipe, x_input, y_true, dl = self.__create_training_pipe(classes=classes,
                                                                      classifier_shape=classifier_shape,
                                                                      classifier_act=classifier_act,
                                                                      num_hidden_layer=num_hidden_layer)

        saver = tf.train.Saver()  # Saver object to store the checkpoints
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) # Activate this line if you want to use only a percentage of GPU memory

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(save_path)
            writer.add_graph(sess.graph)

            for i in range(epoches):
                batches = Batches(x=text, y=one_hot_encodings, batch_size=batch_size)

                for batch in batches:
                    X = batch[0]
                    Y = batch[1]

                    averaged_embeddings_lst = []

                    for paragraph in [word_tokenize(x) for x in X]:
                        word_embeddings = dropout_layer(self.embedding_model.inference(paragraph), wdrop_prob)
                        # Check if Dropout didn't kill every word
                        while word_embeddings.shape[0] == 0:
                            word_embeddings = dropout_layer(self.embedding_model.inference(paragraph),
                                                            dropout_prob=wdrop_prob)
                        # Average Layer
                        averaged_word_embeddings = np.mean(word_embeddings, axis=0)
                        averaged_embeddings_lst.append(averaged_word_embeddings)
                    # start the training
                    train = sess.run(train_pipe,
                                     feed_dict={x_input: np.stack(averaged_embeddings_lst), y_true: np.stack(Y)})

            saver.save(sess, os.path.join(save_path, model_name + ".ckpt"))

            # Save graph definition
            graph = sess.graph
            graph_def = graph.as_graph_def(add_shapes=True)
            tf.train.write_graph(graph_def, ".", os.path.join(save_path, model_name + ".pbtxt"), True)

            # TODO Wrap in a function
            input_graph_path = os.path.join(save_path, model_name + ".pbtxt")
            checkpoint_path = os.path.join(save_path, model_name + ".ckpt")
            input_saver_def_path = ""
            input_binary = False
            output_node_names = "dense_layer_{}/Tanh".format(num_hidden_layer)
            restore_op_name = ""
            filename_tensor_name = ""
            output_frozen_graph_name = os.path.join(save_path, "frozen_graph.pb")
            clear_devices = True

            freeze_graph(input_graph_path, input_saver_def_path,
                         input_binary, checkpoint_path, output_node_names,
                         restore_op_name, filename_tensor_name,
                         output_frozen_graph_name, clear_devices, "")

            self.frozen_graph_path = os.path.abspath(os.path.join(save_path, "frozen_graph.pb"))
            print("model saved to {}.".format(os.path.abspath(save_path)))
            sess.close()
        return None

    def inference(self, text: List[str]) -> np.ndarray:
        """
        Method takes a tokenized sentence as well as the desired output_operation name as parameters

        :param text: Tokenized String
        :type text: list(str)

        :return: the inferred embedding
        :rtype: np.ndarray
        """
        if self.frozen_graph_path == "":
            raise ValueError("Please set the path to the frozen graph.")

        graph = self.initialize_model()

        x = graph.get_tensor_by_name('prefix/placeholder_input:0')
        y = graph.get_operations()[-1].name + ":0"
        with tf.Session(graph=graph) as sess:
            embeddings = self.embedding_model.inference(text)
            dan_embeddings = sess.run(y, feed_dict={x: embeddings})
            sess.close()
            return dan_embeddings

    # TODO Implement a general method
    def inference_batches(self, text: List[List[str]]) -> np.ndarray:
        """
        Method takes a list containing a list with tokenized words.

        :param text: Tokenized String
        :type text: list(str)

        :return: the inferred embedding
        :rtype: np.ndarray
        """
        if self.frozen_graph_path == "":
            raise ValueError("Please set the path to the frozen graph.")

        graph = self.initialize_model()

        x = graph.get_tensor_by_name('prefix/placeholder_input:0')
        y = graph.get_operations()[-1].name + ":0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
        with tf.Session(graph=graph, config=config) as sess:
            embeddings_lst = []
            for paragraph in text:
                embeddings = self.embedding_model.inference(paragraph)
                averaged_embeddings = np.mean(embeddings, axis=0).reshape(1, -1)
                dan_embeddings = sess.run(y, feed_dict={x: averaged_embeddings})
                embeddings_lst.append(dan_embeddings)
            sess.close()
            return embeddings_lst

    def save_config_json(self, config_path: str, **kwargs) -> None:
        for key, value in kwargs.items():
            self.config.update({str(key): str(value)})
        with open(config_path, "w+") as _json:
            json.dump(self.config, _json)

    def set_config(self, json_path: str) -> None:
        with open(json_path) as _json:
            self.config = json.load(_json)
