from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk import RegexpParser
import spacy
from rs_helper.core.Preprocessor import Preprocessor


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
        pos_tagged_text = pos_tag(self.tokens)
        parse_tree = self.parser.parse(pos_tagged_text)
        chunks = self.__get_continuous_chunks(parse_tree)
        chunks = self.__check_dependencies(chunks)
        self.chunks = self.__evaluate_by_position(chunks)
        return self.chunks

    def __evaluate_by_position(self, chunks):
        assigned = list()
        print(chunks)
        for chunk in chunks:
            assigned.append((chunk, self.text.index(chunk)))
        sorted_chunks = sorted(assigned, key=lambda x: x[1])
        return [x[0] for x in sorted_chunks]

    def __check_dependencies(self, chunks: list):
        for i, token in enumerate(self.doc):
            if token.dep_ == "compound":
                if token.text + " " + self.doc[i+1].text not in chunks:
                    chunks.append(token.text + " " + self.doc[i+1].text)
                    if token.text in chunks:
                        chunks.remove(token.text)
                    if self.doc[i+1].text in chunks:
                        chunks.remove(self.doc[i+1].text)
        return chunks
