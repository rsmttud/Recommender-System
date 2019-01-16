from rs_helper.classes import Vectorizer
import numpy as np


class OneHotEncoder(Vectorizer):

    def __init__(self, vocab: list):
        super().__init__()
        self.basis = vocab
        self.text = None

    def vectorize(self, **kwargs):
        """
        :param kwargs: Possible params: text: list(str)
        :return:
        """
        if "text" not in kwargs.keys():
            raise ValueError("Please provide a text that should be encoded")
        self.text = kwargs.get("text")

        def increase(vec, index):
            vec[index] += 1
        vector = np.zeros(len(self.basis))
        [increase(vector, i) for i, w in enumerate(self.basis) if w in self.text]
        return vector
