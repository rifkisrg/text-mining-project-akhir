import Sastrawi

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

factory = StemmerFactory()
stemmer = factory.create_stemmer()

with open("stopwordbahasa.csv", "r") as tala:
    stoplist = tala.read()
    
class preprocess:

    def __init__(self):
        self.file2 = []

    def setText(self, inp):
        self.file2 = inp
    
    def cleaning(docs):
        docs_clean = re.sub("[^a-zA-Z\s]+", " ", docs)
        return docs_clean
    
    def case_folding(docs):
        docs_folding = docs.lower()
        return docs_folding
    
    def tokenization(docs):
        docs_tokening = re.findall("[^\s0-9][A-Za-z]+", docs)
        return docs_tokening
    
    def filtering(docs):
        filtered_sentence = [w for w in docs if not w in stoplist]
        filtered_sentence = []
        for w in docs:
            if w not in stoplist:
                filtered_sentence.append(w)
        return filtered_sentence
    
    def stemming(docs):
        separator = " "
        docs_str = separator.join(docs)
        docs_stemming = stemmer.stem(docs_str)
        docs_stemming_token = re.findall("[^\s0-9][A-Za-z]+", docs_stemming)
        return docs_stemming_token