from rs_helper.classes.Augmentator import Augmentator
from py_translator import Translator


class TranslationAugmentator(Augmentator):

    def __init__(self, sentences: list, languages: list = None):
        """
        :param sentences: List(String): list of sentences to translate
        :param languages: List(String): list of languages ISO-639-1 language codes.
        Class to create augmentations of sentences with Google Translate API
        Langauge codes: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        """
        super().__init__()
        self.translator = Translator()
        if languages is None:
            languages = ["de", "fr", "it", "pl", "pt", "es", "ja"]
        self.target_languages = languages
        self.base_language = "en"
        self.data = sentences
        self.augmented_data = list()

    def run(self):
        """
        :return: List(String)
        Main Controller class
        """
        for sent in self.data:
            self.augmented_data.append(sent)
            translations = self.get_translations(sent, src="en")
            augmentations = self.get_back_translations(translations)
            for el in augmentations:
                self.augmented_data.append(el)
        return self.augmented_data

    def run_2(self):
        augmentations = list()
        for sent in self.data:
            augmentations.append("ORIGINAL:" + sent)
            augmentation = self.get_single_translation(sent)
            augmentations.append(augmentation.text + "\n")
        return augmentations

    def get_single_translation(self, sent: str):
        step_1_trans = self.translator.translate(sent, src="en", dest="kk")
        step_2_trans = self.translator.translate(step_1_trans.text, src="kk", dest="gu")
        return self.translator.translate(step_2_trans.text, src="gu", dest="en")

    def get_translations(self, sent: str, src: str = "en"):
        """
        :param sent: String: sentence to translate
        :param src: Source language key
        :return: Dict(String, String): Dict(Language code, Sentence)
        """
        translations = dict()
        for la in self.target_languages:
            target = self.translator.translate(sent, src=src, dest=la)
            translations[la] = target.text
        return translations

    def get_back_translations(self, translations):
        """
        :param translations: Dict(String, String): Dict(Language code, Sentence)
        :return: List(String): English back translations of sentences
        """
        back_translations = list()
        for la in translations:
            src = self.translator.translate(translations[la], src=la, dest="en")
            back_translations.append(src.text)
        return back_translations
