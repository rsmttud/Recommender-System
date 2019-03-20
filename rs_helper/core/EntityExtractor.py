from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk import RegexpParser
from nltk.corpus import stopwords
from typing import List
import spacy


class EntityExtractor:
    """
    General Class to perform Named Entity Recognition on new problem statements provided in the system.
    Entities will be extracted by the pattern:

        CHUNK: {<NN.*><NN.*>}
        CHUNK: {<DT.*>?<NN.*>}
        CHUNK: {<V.*><N.*>+}

    and then sorted by (Position Index + Term frequency) / n_words
    """
    def __init__(self, text: str) -> None:
        """
        EntityExtractor object to extract analytics entities.

        :param text: The text
        :type text: str
        """
        self.text = text
        self.tokens = word_tokenize(text)
        self.chunk_pattern = r"""
                        CHUNK: {<NN.*><NN.*>}
                        CHUNK: {<DT.*>?<NN.*>}
                        CHUNK: {<V.*><N.*>+}
                    """
        self.parser = RegexpParser(self.chunk_pattern)
        self.chunks = list()
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = list(set(stopwords.words('english')))
        self.doc = self.nlp(self.text)

    def __get_continuous_chunks(self, chunked: Tree) -> List[str]:
        """
        Transforms a NLTK ParseTree to a list of chunks as strings

        :param chunked: Tree after initial parsing the sentence with the given pattern
        :type chunked: Tree

        :return: list of found chunks
        :rtype: list(str)
        """
        continuous_chunk = []
        current_chunk = []
        for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves() if pos != "DT"]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                current_chunk = []
        if len(current_chunk) > 0:
            continuous_chunk.append(" ".join(current_chunk))
        return continuous_chunk

    def extract_entities(self) -> List[str]:
        """
        Generel handler - Methods extracts the chunks of a sentence by the pattern in self.pattern

        :return: List of chunks
        :rtype: list(str)
        """
        pos_tagged_text = pos_tag(self.tokens)
        parse_tree = self.parser.parse(pos_tagged_text)
        chunks = self.__get_continuous_chunks(parse_tree)
        chunks = self.__check_dependencies(chunks)
        self.chunks = self.__score_chunks(chunks)
        return self.chunks

    def __score_chunks(self, chunks: List[str]) -> List[str]:
        """
        Method evaluates the supplied chunks by calculating their score with
        (Position Index + Term frequency) / n_words

        :param chunks: Unscored chunks
        :type chunks: list(str)

        :return: Scored chunks
        :rtype: list(str)
        """
        assigned = list()
        haystack = " ".join([x for x in self.tokens if x not in self.stopwords])
        print(haystack)
        for chunk in chunks:
            try:
                index = haystack.index(chunk)
            except ValueError:
                if chunk.find(" ") != -1:
                    index = haystack.index(chunk.split(" ")[1])
                else:
                    index = 0
            N = len(self.tokens)
            tf = self.text.count(chunk)
            assigned.append((chunk, (index+tf)/N))  # Calculate the scoring mechanism
        sorted_chunks = sorted(assigned, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_chunks]

    def __check_dependencies(self, chunks: List[str]) -> List[str]:
        """
        Method extends existing chunks by their potentially compount-related words

        :param chunks: List of Chunks
        :type chunks: list(str)

        :return: Updated chunks
        :rtype: list(str)
        """
        for i, token in enumerate(self.doc):
            if token.dep_ == "compound":
                if token.text + " " + self.doc[i+1].text not in chunks:
                    chunks.append(token.text + " " + self.doc[i+1].text)
                    if token.text in chunks:
                        chunks.remove(token.text)
                    if self.doc[i+1].text in chunks:
                        chunks.remove(self.doc[i+1].text)
        return chunks
