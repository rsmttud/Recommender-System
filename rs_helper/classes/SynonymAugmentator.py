from rs_helper.classes.Augmentator import Augmentator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import random


class SynonymAugmentator(Augmentator):

    def __init__(self, sentences: list, n_augmentations: int = 3):
        """
        :param sentences: List(String) (Sentences to get augmented)
        :param n_augmentations: int (Number of augmentations per sentence)
        """
        super().__init__()
        self.sentences = sentences
        self.n_augmentations = n_augmentations

    def augment(self) -> list:
        """
        :return: List(String)
        Method to replace all nouns, verbs and adverbs by synonyms
        """
        augmentations = list()
        for s in self.sentences:
            words = word_tokenize(s)
            pos_tags = nltk.pos_tag(words)
            for i in range(self.n_augmentations):
                for i, (w, p) in enumerate(pos_tags):
                    r = random.random()
                    if r > 0.5:
                        if p in ["NN", "NNP", "NNS", "VBZ", "VB", "VBD", "VBG", "VBN", "VBP", "JJ", "JJR", "JJS"]:
                            w_sense = lesk(words, w, self.get_wordnet_pos(p))
                            if not w_sense:
                                w_sense = w
                            synonyms = self.__get_synonyms(w_sense)
                            synonyms = self.__clean_synonyms(synonyms)
                            r = random.randrange(0, len(synonyms))
                            words[i] = synonyms[r]
                augmentations.append(" ".join(words))
        return augmentations

    # TODO If the word was not found - so no synset was found - take the first synset of the word
    def __get_synonyms(self, word) -> list:
        """
        :param word: String
        :return: list (Synonyms of the word)
        """
        if isinstance(word, str):
            word = wn.synsets(word)[0]
        return [l.name() for l in word.lemmas()]

    def __clean_synonyms(self, synonyms: list) -> list:
        """
        :param synonyms: list(String)
        :return: list
        Removes underscores from synonyms.
        """
        return [syn.replace("_", " ") for syn in synonyms]

    @staticmethod
    def get_wordnet_pos(pos_tag: str):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN  # Noun is default in lemmatizer


if __name__ == "__main__":
    sents = ["In cluster analysis a large number of methods are available for classifying objects "
             "on the basis of their dissimilarities and similarities",
             "Cluster analysis is a multivariate method which aims to classify a sample of subjects "
             "or objects on the basis of a set of measured variables into a number of different "
             "groups such that similar subjects are placed in the same group"]
    s = SynonymAugmentator(sentences=sents)
    augs = s.augment()
    for a in augs:
        print(a)