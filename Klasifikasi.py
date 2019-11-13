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
		self.train_res = []
        
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
		weight.getFeatures()
		weight.getTF()
		weight.getIDF()
		weight.getTFIDF()
		result_weight = weight.getNormal()
		
		avg_term_class = []
		
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
			
			avg_term_class.append([avg_finance, avg_food, avg_health, avg_inet, avg_sport])
		
		self.train_res = avg_term_class
		
	def test(self, files):
		tokens = []
		result_weight = []
		
		# preprocessing
		for file in files:
			docs_cleaned = pp.cleaning(file)
			docs_folded = pp.case_folding(docs_cleaned)
			docs_token = pp.tokenization(docs_folded)
			docs_filtered = pp.filtering(docs_token)
			docs_stemming_token = pp.stemming(docs_filtered)
			tokens.append(docs_stemming_token)
		
		# weighting
		for document in tokens:
			print(len(document))
			weight = Weight()
			weight.setText([document])
			weight.getFeatures()
			weight.getTF()
			weight.getIDF()
			result_weight.append(weight.getTFIDF())
			
		print(result_weight)

	def hitungAkurasi(self, kelas_hasil, kelas_asli):
		return 0
