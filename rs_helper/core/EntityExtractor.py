from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk import RegexpParser
import spacy


class EntityExtractor:

    def __init__(self, text: str):
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
        self.doc = self.nlp(self.text)

    def __get_continuous_chunks(self, chunked):
        """
        :param chunked: NLTK ParseTree - Tree after initial parsing the sentence with the given pattern
        :return: List(String)
        Transforms a NLTK ParseTree to a list of chunks as strings
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
                else:
                    continue
        if len(current_chunk) > 0:
            continuous_chunk.append(" ".join(current_chunk))
        return continuous_chunk

    def extract_entities(self):
        """
        :return: List(String) - List of chunks
        Generel handler - Methods extracts the chunks of a sentence by the pattern in self.pattern
        """
        pos_tagged_text = pos_tag(self.tokens)
        parse_tree = self.parser.parse(pos_tagged_text)
        chunks = self.__get_continuous_chunks(parse_tree)
        chunks = self.__check_dependencies(chunks)
        self.chunks = self.__score_chunks(chunks)
        return self.chunks

    def __score_chunks(self, chunks):
        """
        :param chunks: List(String) - Unscored chunks
        :return: List(String) - Scored chunks
        Method evaluates the supplied chunks by calculating their score with
        (Position Index + Term frequency) / n_words
        """
        assigned = list()
        # print(chunks)
        for chunk in chunks:
            index = self.text.index(chunk)
            N = len(self.tokens)
            tf = self.text.count(chunk)
            assigned.append((chunk, (index+tf)/N))  # Calulate the scoring mechanism
        sorted_chunks = sorted(assigned, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_chunks]

    def __check_dependencies(self, chunks: list):
        """
        :param chunks: List(String) - List of Chunks
        :return: List(String) - Updated chunks
        Method extends existing chunks by their potentially compount-related words
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
