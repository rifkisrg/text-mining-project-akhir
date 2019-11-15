import os
import glob
import errno
from Preprocessing import preprocess as pp
from Weighting import Weight
import pandas as pd
from statistics import mean
import pickle
import numpy as np

class Klasifikasi:
	def __init__(self):
		self.files_name = []
		self.files_category = []
		self.files = []
		

		f = open('dataTrain.csv', 'rb')
		self.train_res = pickle.load(f)
		f.close()

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
		
	# def train(self, files, category):
	# 	self.files_category = category
	# 	tokens = []

	# 	# preprocessing
	# 	for file in files:
	# 		docs_cleaned = pp.cleaning(file)
	# 		docs_folded = pp.case_folding(docs_cleaned)
	# 		docs_token = pp.tokenization(docs_folded)
	# 		docs_filtered = pp.filtering(docs_token)
	# 		docs_stemming_token = pp.stemming(docs_filtered)
	# 		tokens.append(docs_stemming_token)

	# 	# weighting
	# 	weight = Weight()
	# 	weight.setText(tokens)

	# 	f = open('fiturDataTrain.csv', 'wb')
	# 	pickle.dump(weight.getFeatures(), f)
	# 	f.close()
		
	# 	weight.getTF()
	# 	weight.getIDF()
	# 	weight.getTFIDF()
	# 	result_weight = weight.getNormal()
		
	# 	avg_term_class = []
		
	# 	for term in result_weight:
	# 		sum_finance = 0
	# 		count_finance = 0
	# 		sum_food = 0
	# 		count_food = 0
	# 		sum_health = 0
	# 		count_health = 0
	# 		sum_inet = 0
	# 		count_inet = 0
	# 		sum_sport = 0
	# 		count_sport = 0
			
	# 		for i in range(len(term)):
	# 			if (self.files_category[i] == "finance"):
	# 				sum_finance += term[i]
	# 				count_finance += 1
	# 			elif (self.files_category[i] == "food"):
	# 				sum_food += term[i]
	# 				count_food += 1
	# 			elif (self.files_category[i] == "health"):
	# 				sum_health += term[i]
	# 				count_health += 1
	# 			elif (self.files_category[i] == "inet"):
	# 				sum_inet += term[i]
	# 				count_inet += 1
	# 			elif (self.files_category[i] == "sport"):
	# 				sum_sport += term[i]
	# 				count_sport += 1
			
	# 		avg_finance = sum_finance / count_finance
	# 		avg_food = sum_food / count_food
	# 		avg_health = sum_health / count_health
	# 		avg_inet = sum_inet / count_inet
	# 		avg_sport = sum_sport / count_sport
			
	# 		avg_term_class.append([avg_finance, avg_food, avg_health, avg_inet, avg_sport])

		# file_train = 'dataTrain.csv'
		# f = open(file_train, 'wb')
		# pickle.dump(avg_term_class, f)
		# f.close()
	
	def test(self):
		# tokens = []

		# for file in files:
		# 	docs_cleaned = pp.cleaning(file)
		# 	docs_folded = pp.case_folding(docs_cleaned)
		# 	docs_token = pp.tokenization(docs_folded)
		# 	docs_filtered = pp.filtering(docs_token)
		# 	docs_stemming_token = pp.stemming(docs_filtered)
		# 	tokens.append(docs_stemming_token)

		# weight = Weight()
		# weight.setText(tokens)
		# weight.getFeatures()
		# weight.getTF()
		# weight.getIDF()
		# weight.getTFIDF()

		f1 = open('fiturDataTrain.csv', 'rb')
		fiturDataTrain = pickle.load(f1)
		f1.close()
		
		f2 = open('fiturDataUji.csv', 'rb')
		fiturDataUji = pickle.load(f2)
		f2.close()

		f3 = open('dataTrain.csv', 'rb')
		dataTrain = pickle.load(f3)
		f3.close()

		f4 = open('dataUji.csv', 'rb')
		dataUji = pickle.load(f4)
		f4.close()

		# xa = [list(item) for item in zip(dataUji, fiturDataUji)]

		newFiturDataUji = []
		newDataUji = []
		# newFeat = []

		for x in range(0, len(fiturDataTrain)):
			for y in range(0, len(fiturDataUji)):
				if fiturDataTrain[x] == fiturDataUji[y]:
					newFiturDataUji.append(fiturDataUji[y])
					newDataUji.append(dataUji[y])
			if fiturDataTrain[x] not in newFiturDataUji:
				newDataUji.append([0.0] * len(dataUji[0]))

		dataUji_zipped = [list(item) for item in zip(*newDataUji)]
		dataTrain_zipped = [list(item) for item in zip(*dataTrain)]
		# x = [fiturDataTrain, newFeatDataUji]

		dump = []
		for x in dataUji_zipped:
			zat = []
			for y in dataTrain_zipped:
				test = [np.array(x) * np.array(y)]
				zat.append(sum(test))
			dump.append(zat)

		haha = []
		for x in dump:
			yoi = []
			for y in x:
				new_dump = sum(y)
				yoi.append(new_dump)
			haha.append(yoi)
			yoi = []

		cat = ['finance', 'food', 'health', 'inet', 'sport']
		hehe = []
		for x in haha:
			ind = np.argmax(x)
			hehe.append(cat[ind])

		# print(pd.DataFrame(newFeatDataUji))
		# print(dump)
		# print(pd.DataFrame(dump))
		print(pd.DataFrame(hehe))
		# print(pd.DataFrame(newDataUji))
		# print(pd.DataFrame(dataTrain, index = fiturDataTrain))
		# print(pd.DataFrame(fiturDataTrain))
		# print(pd.DataFrame(newFeatDataUji))

	# test_normal_tpose = [list(item) for item in zip(*test_normal)]
		# train_tpose = [list(item) for item in zip(*self.train_res)]
		
		# table123 = []

		# for docs in test_normal_tpose:
		# 	for kelas in train_tpose:
		# 		dummy =  [np.array(docs) * np.array(kelas)]
		# 	table123.append(dummy)

		
		# for test_docs in test_normal_tpose:
		# 	for x in range(inc, len(test_docs)):
		# 		for train_class in self.train_res:
		# 			dummy = [test_docs[x] * i for i in train_class]
		# 		table123.append(sum(dummy))
		# 		inc += 1
		# 		dummy = []
		# temp = []
		# for docs in test_normal:
		# 	for term in docs:
		# 		for x in range(0, len(self.train_res)):
		# 			dummy = [term * i for i in self.train_res[x]]
		# 		avg = sum(dummy)
		# 	table123.append(avg)
			# for x in range(0, len(self.train_res)):
			# 	for y in range(0, len(docs)):
			# 		dummy = [docs[y] * i for i in self.train_res[x]]
			# 	temp.append(sum(dummy))
			# table123.append(temp)

		# print(pd.DataFrame(train_tpose))
		# print(pd.DataFrame(test_normal_tpose))
		# print(pd.DataFrame(feat))

	def hitungAkurasi(self, kelas_hasil, kelas_asli):
		return 0
