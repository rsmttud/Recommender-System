import os
import re


class Corpora:
    """
    This class contains the current process descriptions and the referred label
    """

    def __init__(self, path: str, label: str = None):
        """
        :param path: The path to the directory where all the .txt are stored or an path to an txt-file
        :param label: A category to mark the text
        """
        if not isinstance(path, str):
            raise ValueError("Parameter path must be string.")

        self.path = path
        self.n_words: int = 0
        self.n_sentences: int = 0
        self.label: str = label
        self.data = ""

        self.__read_text()
        self.__count_words()
        self.__delete_multiple_spaces()

    def __read_text(self):
        # Reading all txt from a dict
        if os.path.isdir(self.path):
            for txt in [x for x in os.listdir(self.path) if x.endswith((".txt", ".TXT", ".Txt"))]:
                file = open(os.path.join(self.path, txt), "r")
                text = file.read()
                if self.data == "":
                    self.data += text
                else:
                    self.data += " " + text
                file.close()

        # Reading all txt from a file
        elif self.path.endswith((".txt", ".TXT", ".Txt")):
            with open(self.path, "r") as file:
                self.data = file.read()

    def __count_words(self, include_duplicates=True) -> None:
        self.n_words = len(self.data.split())

    def __count_sentences(self) -> None:
        # TODO: Using a Sentence Tokenizer and Count? Is a waste of computational power just for a number.
        pass

    def __delete_multiple_spaces(self) -> None:
        self.data = re.sub(' +', ' ', self.data)

    def save(self, out_path: str) -> bool:
        """
        This method saves the supplied problem description to a .txt file
        :param out_path: full path to the out_file
        :return: boolean
        """
        # With Block is an Try/Except -Block - redundant
        try:
            file = open(out_path, "w")
            file.write(self.data)
            file.close()
            return True
        except:
            return False
