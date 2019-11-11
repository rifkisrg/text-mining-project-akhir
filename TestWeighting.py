from Weighting import Weight
from Preprocessing import preprocess as pp
import pandas as pd

with open("test1.txt", "r") as myfile:
    datatest = myfile.read()
with open("test2.txt", "r") as myfile2:
    datatest2 = myfile2.read()
with open("test3.txt", "r") as myfile3:
    datatest3 = myfile3.read()


arr = [datatest, datatest2, datatest3]
arr2 = []
for docs in arr:
    docs_cleaned = pp.cleaning(docs)
    docs_folded = pp.case_folding(docs_cleaned)
    docs_token = pp.tokenization(docs_folded)
    docs_filtered = pp.filtering(docs_token)
    docs_stemming_token = pp.stemming(docs_filtered)
    arr2.append(docs_stemming_token)

weighting = Weight()
weighting.setText(arr2)
print(weighting.getFeatures())
# weighting.getTF()
print(pd.DataFrame(weighting.getTF(), index = weighting.getFeatures(), columns = ["test1","test2","test3"]))
print(pd.DataFrame(weighting.getIDF(), index = weighting.getFeatures(), columns = ["IDF"]))
print(pd.DataFrame(weighting.getTFIDF(), index = weighting.getFeatures(), columns = ["test1","test2","test3"]))
print(weighting.getNormal())