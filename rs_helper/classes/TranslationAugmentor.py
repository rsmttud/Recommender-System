from rs_helper.classes.Augmentator import Augmentator
from py_translator import Translator


class TranslationAugmentator(Augmentator):

    def __init__(self, sentences: list):
        """
        :param sentences: List(String): list of sentences to translate
        :param languages: List(String): list of languages ISO-639-1 language codes.
        Class to create augmentations of sentences with Google Translate API
        Langauge codes: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        """
        super().__init__()
        self.translator = Translator()
        self.data = sentences
        self.augmented_data = list()

    def run(self):
        augmentations = list()
        for sent in self.data:
            augmentations.append("ORIGINAL:" + sent)
            augmentation = self.translation_loop(sent)
            augmentations.append(augmentation.text + "\n")
        return augmentations

    def translation_loop(self, sent: str):
        print(sent)
        step_1_trans = self.translator.translate(sent, src="en", dest="am")
        step_2_trans = self.translator.translate(step_1_trans.text, src="am", dest="gu")
        return self.translator.translate(step_2_trans.text, src="gu", dest="en")
