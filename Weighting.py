import math
import numpy as np

class Weight:
    def __init__(self):
        self.file2 = []
        self.temp = []
        self.TF = []
        self.IDF = []
        self.TFIDF = []
        self.normal = []

    def setText(self, inp):
        self.file2 = inp

    def getFeatures(self):
        for x in self.file2:
            for words in x:
                if words not in self.temp:
                    self.temp.append(words)

        return self.temp

    def getTF(self):
        for feature in self.temp:
            document_count = []
            for document in self.file2:
                if document.count(feature) == 0:
                    document_count.append(document.count(feature))
                else:
                    ct = 1 + math.log10(document.count(feature))
                    document_count.append(ct)

            self.TF.append(document_count)

        return self.TF

    def getIDF(self):
        q = 0
        DF = []
        leng = len(self.file2)

        for x in self.TF:
            for y in x:
                if y != 0:
                    q += 1
            DF.append(q)
            q = 0

        for z in DF:
            s = math.log10(leng/z)
            self.IDF.append(s)

        return self.IDF

    def getTFIDF(self):
        temp = []
        inc = 0

        for i in self.IDF:
            for j in range(inc, len(self.TF)):
                for k in self.TF[j]:
                    res = i * k

                    temp.append(res)
                self.TFIDF.append(temp)
                temp = []
                inc+=1
                break
        
        return self.TFIDF

    def getNormal(self):
        normalTemp = []
        temp = []
        sm = 0
        tpose = tuple(zip(*self.TFIDF))

        # for row in tpose:
        for x in range(0, len(tpose)):
            for i in tpose[x]:
                sm += math.pow(i, 2)
                sms = math.sqrt(sm)
            normalTemp.append(sms)
            sm = 0

        for row in self.TFIDF:
            for i in range(0, len(row)):
                total = row[i] / normalTemp[i]
                temp.append(total)
            self.normal.append(temp)
            temp = []


        return self.normal

    def getAvg(self):
        temp = []
        avgs = []
        transpose = tuple(zip(*self.normal))

        for x in range(0, len(transpose)):
            avg = sum(transpose[x]) / len(transpose[x])
            temp.append(avg)

        
        avg = sum(temp) / len(temp)
        avgs.append(avg)
            
        return avgs
    