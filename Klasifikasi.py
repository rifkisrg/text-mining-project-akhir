import os
import glob
import errno
from Preprocessing import preprocess as pp
from Weighting import Weight
import pandas as pd
from statistics import mean


class Klasifikasi:
    def __init__(self):
        self.files_name = []
        self.files_category = []
        self.files = []
        self.avg_class = []

    def getFiles(self, path):
        files_in_path = glob.glob(path)

        files = []

        for file in files_in_path:
            try:
                with open(file) as f:
                    x = f.read()
                    files.append(x)
            except IOError as exc:
                    if exc.errno != errno.EISDIR:
                        raise

        return files

    def train(self, files, category):
        tokens = []
        temp = []
        temp2 = []

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
        term_f = weight.getTFIDF()
        # weight.getNormal()
        # docs_with_class = [list(item) for item in zip(weight.getAvg(), category)]

        # for cat in category:
        #     for x in range(0, len(docs_with_class)):
        #         if cat == docs_with_class[x][1]:
        #             temp.append(docs_with_class[x][0])
        #     avg = mean(temp)
        #     if avg not in self.avg_class:
        #         self.avg_class.append(avg)
        #     temp = []

        # tf_tpose = [list(row) for row in zip(*weight.getTF())]
        # tf_with_class = [list(item) for item in zip(tf_tpose, category)]

        

        for row in term_f:
            zipped = [list(item) for item in zip(row, category)]
            temp.append(zipped)

        list_avg = []
        for row in temp:
            for cat in category:
                for terms in row:
                    if cat == terms[1]:
                        temp2.append(terms[0])
                    else:
                        break
                avg = mean(temp2)
                list_avg.append(avg)
            self.avg_class.append(list_avg)
            temp2 = []
            list_avg = []

        # print(pd.DataFrame(temp))
        print(pd.DataFrame(self.avg_class))
        # print(pd.DataFrame(tf_with_class))

    # def test(self, files):
    #     tokens = []

        # for file in files:
        #     docs_cleaned = pp.cleaning(file)
        #     docs_folded = pp.case_folding(docs_cleaned)
        #     docs_token = pp.tokenization(docs_folded)
        #     docs_filtered = pp.filtering(docs_token)
        #     docs_stemming_token = pp.stemming(docs_filtered)
        #     tokens.append(docs_stemming_token)

        # weight = Weight()
        # weight.setText(tokens)
        # weight.getFeatures()
        # # weight.getTF()
        # # weight.getIDF()
        # # weight.getTFIDF()
        # print(pd.DataFrame(weight.getTF()))