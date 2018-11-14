from rs_helper.classes.Keyword import Keyword


class Topic:

    def __init__(self, class_name: str):
        """
        :param class_name: String
        Keyword set is created automatically
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
                                          x.algorithm + self.get_keyword_algorithm(other, x.keyword))
                else:
                    topic_set.add_keyword(x.keyword, x.rank, x.algorithm)
        return topic_set

    def __len__(self):
        return len(self.keyword_set)

    def __iter__(self):
        return iter(self.keyword_set)

    def sort_by_rank(self):
        self.keyword_set.sort(key=lambda x: x.rank, reverse=True)

    def norm_ranks(self):
        maximum = max([x.rank for x in self.keyword_set])
        minimum = min([x.rank for x in self.keyword_set])
        normed_keyword_set = []
        for keyword in self.keyword_set:
            try:
                keyword.rank = (keyword.rank-minimum) / (maximum-minimum)
            except ZeroDivisionError as e:
                print(e)
            normed_keyword_set.append(keyword)
        self.keyword_set = normed_keyword_set

    def add_keyword(self, keyword, rank: float, algorithm: str) -> None:
        self.keyword_set.append(Keyword(keyword, rank, algorithm))

    def get_keywords(self, duplicates=True) -> list:
        if duplicates:
            return self.keyword_set
        else:
            return list(set(self.keyword_set))

    def pretty_print(self, duplicates=True):
        print("________TopicSet - {}________".format(self.class_name))
        if duplicates:
            for keyword in self.keyword_set:
                print(keyword.__str__())
        else:
            for keyword in set(self.keyword_set):
                print(keyword.__str__())

    def get_keyword_names(self):
        return [x.keyword for x in self.keyword_set]

    def get_keyword_rank(self, topic, key):
        return next((x.rank for x in topic.keyword_set if x.keyword == key), None)

    def get_keyword_algorithm(self, topic, key):
        return next((x.algorithm for x in topic.keyword_set if x.keyword == key), None)