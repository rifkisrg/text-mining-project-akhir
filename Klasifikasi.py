import os
import glob
import errno
from Preprocessing import preprocess as pp
from Weighting import Weight
import pandas as pd
from statistics import mean
import itertools

class Klasifikasi:
    def __init__(self):
        self.files_name = []
        self.files_category = []
        self.files = []

    def train(self, files, category):
        tokens = []
        avg_class = []

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
        weight.getNormal()
        docs_with_class = [list(item) for item in zip(weight.getAvg(), category)]

        for x in range(0, len(docs_with_class)):
            temp = []
            if docs_with_class[x][1] in category:
                temp.append(docs_with_class[x][0])
                avg = mean(temp)
                avg_class.append(avg)
                temp = []

        print(pd.DataFrame(avg_class))

# weight.setText(documents)
# feat = weight.getFeatures()
# weight.getTF()
# weight.getIDF()
# weight.getTFIDF()
# print(pd.DataFrame(weight.getNormal()))