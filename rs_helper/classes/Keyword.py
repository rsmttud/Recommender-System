

class Keyword:
    def __init__(self, keyword: list, rank: float, algorithm: str = None):
        """
        :param keyword: List(String)
        :param rank: float
        :param algorithm: String
        """
        self.keyword = keyword if isinstance(keyword, list) else [keyword]
        self.rank = rank
        self.algorithm = algorithm

    def __str__(self):
        return str(self.keyword) + " | " + str(self.rank) + " | " + str(self.algorithm)

    def __repr__(self):
        return "({}, {})".format(self.keyword, self.rank)

    def __eq__(self, other):
        if self.keyword == other.keyword:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.keyword)