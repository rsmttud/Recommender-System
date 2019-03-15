import os
import re


class Corpora:
    """
    This class contains the current process descriptions and the referred label
    """

    def __init__(self, path: str, label: str = None):
        """
        General corpora class.

        :param path: The path to the directory where all the .txt are stored or an path to an txt-file
        :type path: str

        :param label: A category to mark the text
        :type label: str
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

    def __read_text(self) -> None:
        """
        The methods reads text files and stores them into the data class attribute. Depending on the given path in
        the constructor it distinguishes between directories or a single file. In case of an directory all txt files are
        stored in the data attribute.

        :return: None
        """
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
        """
        Count the words across the corpus. Count will be stored in class attribute n_words

        :param include_duplicates: Weather duplicated should be included in counting
        :type include_duplicates: bool

        :return: None
        """
        self.n_words = len(self.data.split())

    def __delete_multiple_spaces(self) -> None:
        """
        Delete duplicated spaces from data.

        :return: None
        """
        self.data = re.sub(' +', ' ', self.data)

    def save(self, out_path: str) -> bool:
        """
        This method saves the supplied problem description to a .txt file

        :param out_path: full path to the out_file
        :type out_path: str

        :return: Boolean if save worked
        :rtype: bool
        """
        try:
            file = open(out_path, "w")
            file.write(self.data)
            file.close()
            return True
        except:
            return False
