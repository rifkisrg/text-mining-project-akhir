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
		self.fiturDataTrain = []
		self.files_new_category = []
		self.files = []
		self.avg_term_class = []

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

	def getCategories(self, path):
		kelas_asli = []
		files_in_path = glob.glob(path)
		for file in files_in_path:
			kelas_asli.append(os.path.split(os.path.dirname(file))[-1])
		
		return kelas_asli

	def train(self, files, category):
		self.files_category = category
		tokens = []

		# preprocessing
		for file in files:
			docs_cleaned = pp.cleaning(file)
			docs_folded = pp.case_folding(docs_cleaned)
			docs_token = pp.tokenization(docs_folded)
			docs_filtered = pp.filtering(docs_token)
			docs_stemming_token = pp.stemming(docs_filtered)
			tokens.append(docs_stemming_token)

		# weighting
		weight = Weight()
		weight.setText(tokens)
		self.fiturDataTrain = weight.getFeatures()
		weight.getTF()
		weight.getIDF()
		weight.getTFIDF()
		result_weight = weight.getNormal()
		
		for term in result_weight:
			sum_finance = 0
			count_finance = 0
			sum_food = 0
			count_food = 0
			sum_health = 0
			count_health = 0
			sum_inet = 0
			count_inet = 0
			sum_sport = 0
			count_sport = 0
			
			for i in range(len(term)):
				if (self.files_category[i] == "finance"):
					sum_finance += term[i]
					count_finance += 1
				elif (self.files_category[i] == "food"):
					sum_food += term[i]
					count_food += 1
				elif (self.files_category[i] == "health"):
					sum_health += term[i]
					count_health += 1
				elif (self.files_category[i] == "inet"):
					sum_inet += term[i]
					count_inet += 1
				elif (self.files_category[i] == "sport"):
					sum_sport += term[i]
					count_sport += 1
			
			avg_finance = sum_finance / count_finance
			avg_food = sum_food / count_food
			avg_health = sum_health / count_health
			avg_inet = sum_inet / count_inet
			avg_sport = sum_sport / count_sport
			
			self.avg_term_class.append([avg_finance, avg_food, avg_health, avg_inet, avg_sport])
	
	def test(self, files):
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
		fiturDataUji = weight.getFeatures()
		weight.getTF()
		weight.getIDF()
		weight.getTFIDF()
		dataUji = weight.getNormal()
		newFiturDataUji = []
		newDataUji = []

		for x in range(0, len(self.fiturDataTrain)):
			for y in range(0, len(fiturDataUji)):
				if self.fiturDataTrain[x] == fiturDataUji[y]:
					newFiturDataUji.append(fiturDataUji[y])
					newDataUji.append(dataUji[y])
			if self.fiturDataTrain[x] not in newFiturDataUji:
				newDataUji.append([0.0] * len(dataUji[0]))

		dataUji_zipped = [list(item) for item in zip(*newDataUji)]
		dataTrain_zipped = [list(item) for item in zip(*self.avg_term_class)]

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
		kelas_hasil = []
		for x in haha:
			ind = np.argmax(x)
			kelas_hasil.append(cat[ind])

		return kelas_hasil

	def hitungAkurasi(self, kelas_hasil, kelas_asli):
		temp = 0

		for x in range(0, len(kelas_hasil)):
			if kelas_hasil[x] == kelas_asli[x]:
				temp += 1

		accuration = (temp / len(kelas_hasil)) * 100

		return accuration