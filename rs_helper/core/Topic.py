from rs_helper.core.Keyword import Keyword
from typing import *


class Topic:
    def __init__(self, class_name: str) -> None:
        """
        Keyword set is created automatically

        :param class_name: Name of the class
        :type class_name: str
        """
        self.class_name = class_name
        self.keyword_set = []

    def __repr__(self):
        return "class: {} \nkeywords: {}".format(self.class_name, self.keyword_set)

    def __add__(self, other):
        if self.class_name != other.class_name:
            raise ValueError("The topics belong to different classes and should not be merged")

        a_keyword_names = self.get_keyword_names()
        b_keyword_names = other.get_keyword_names()
        topic_set = Topic(class_name=self.class_name)
        merged = self.keyword_set + other.keyword_set
        for x in merged:
            if x not in topic_set.keyword_set:
                if x.keyword in a_keyword_names and x.keyword in b_keyword_names:
                    topic_set.add_keyword(x.keyword,
                                          x.rank + self.get_keyword_rank(other, x.keyword),
                                          x.algorithm + " " + self.get_keyword_algorithm(other, x.keyword))
                else:
                    topic_set.add_keyword(x.keyword, x.rank, x.algorithm)
        return topic_set

    def __len__(self):
        return len(self.keyword_set)

    def __iter__(self):
        return iter(self.keyword_set)

    def sort_by_rank(self) -> None:
        """
        Sort the topic keywords by their rank

        :return: None
        """
        self.keyword_set.sort(key=lambda x: x.rank, reverse=True)

    def norm_ranks(self) -> None:
        """
        Normalize the keyword ranks by Min-Max-Normalization

        :return: None
        """
        maximum = max([x.rank for x in self.keyword_set])
        minimum = min([x.rank for x in self.keyword_set])
        normed_keyword_set = []
        for keyword in self.keyword_set:
            try:
                keyword.rank = (keyword.rank - minimum) / (maximum - minimum)
            except ZeroDivisionError as e:
                print(e)
            normed_keyword_set.append(keyword)
        self.keyword_set = normed_keyword_set

    def add_keyword(self, keyword: List[str], rank: float, algorithm: str) -> None:
        """
        Add another keyword to the topic

        :param keyword: The tokenized keyword
        :type keyword: list(str)
        :param rank: The keywords rank
        :type rank: float
        :param algorithm: The algorithm that was used for extraction
        :type algorithm: str

        :return: None
        """
        self.keyword_set.append(Keyword(keyword, rank, algorithm))

    def get_keywords(self, duplicates: bool = True) -> List[Keyword]:
        """
        Get the list of unique keywords

        :param duplicates: Include duplicated in the list
        :type duplicates: bool

        :return: List of keywords
        :rtype: list(Keyword)
        """
        if duplicates:
            return self.keyword_set
        else:
            return list(set(self.keyword_set))

    def pretty_print(self, duplicates: bool = True) -> None:
        """
        Print out the topic pretty formatted

        :param duplicates: Include duplicates?
        :type duplicates: bool

        :return: None
        """
        print("________TopicSet - {}________".format(self.class_name))
        if duplicates:
            for keyword in self.keyword_set:
                print(keyword.__str__())
        else:
            for keyword in set(self.keyword_set):
                print(keyword.__str__())

    def get_keyword_names(self) -> List[str]:
        """
        Get a list of the keyword names only

        :return: list of keyword names
        :rtype: list(list(str))
        """
        return [x.keyword for x in self.keyword_set]

    def get_keyword_rank(self, topic, key) -> List[int]:
        """
        Get the rank of a specific keyword

        :param topic: Topic object to receive rank of word from
        :type topic: Topic
        :param key: The keyword to get the rank from
        :type key: list(str)

        :return: The keywords rank
        :rtype: float
        """
        return next((x.rank for x in topic.keyword_set if x.keyword == key), None)

    def get_keyword_algorithm(self, topic, key: str) -> List[str]:
        """
        Get the algorithm of a specific keyword

        :param topic: Topic object to receive algorithm of word from
        :type topic: Topic
        :param key: The keyword to get the algorithm from
        :type key: list(str)

        :return: The keywords algorithm
        :rtype: str
        """
        return next((x.algorithm for x in topic.keyword_set if x.keyword == key), None)
