import os
import glob
import errno
from Preprocessing import preprocess as pp
from Weighting import Weight
import pandas as pd

class Klasifikasi:
    def __init__(self):
        self.files_name = []
        self.files_category = []
        self.files = []

    def train(self, files, category):
        tokens = []

        for file in files:
            docs_cleaned = pp.cleaning(file)
            docs_folded = pp.case_folding(docs_cleaned)
            docs_token = pp.tokenization(docs_folded)
            docs_filtered = pp.filtering(docs_token)
            docs_stemming_token = pp.stemming(docs_filtered)
            tokens.append(docs_stemming_token)

        weight = Weight()
        weight.setText(tokens)
        weight.getFeatures()
        weight.getTF()
        weight.getIDF()
        weight.getTFIDF()
        print(pd.DataFrame(zip(weight.getNormal(), category)))
        

# weight.setText(documents)
# feat = weight.getFeatures()
# weight.getTF()
# weight.getIDF()
# weight.getTFIDF()
# print(pd.DataFrame(weight.getNormal()))