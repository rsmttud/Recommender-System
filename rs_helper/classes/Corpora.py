
class Corpora:
    """
    This class contains the current process descriptions and the referred label
    """
    def __init__(self, path: str, label: str = None):
        """
        :param path: The process description as a normal String
        """
        if not isinstance(path, str):
            raise ValueError("Parameter path must be string.")

        self.n_words: int = 0
        # TODO: n_documents macht nach dem jetzigen stand ja keinen Sinn mehr -> Ist ja immer nur eine Problemstellung
        self.n_documents: int = 0

        self.label: str = label
        with open(path, "r") as file:
            self.data = file.read()

    def save(self, out_path: str) -> bool:
        """
        This method saves the supplied problem description to a .txt file
        :param out_path: full path to the out_file
        :return: boolean
        """
        try:
            with open(out_path, "w") as file:
                file.write(self.data)
        except:
            return False
        return True
