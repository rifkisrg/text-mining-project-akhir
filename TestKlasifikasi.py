from Klasifikasi1 import Klasifikasi
import os
import glob
import errno
import datetime

print("Start = ", datetime.datetime.now())
path = input('Input Direktori: ')
path_data_uji = input("Input Data Uji: ")

klasifikasi = Klasifikasi()
klasifikasi.train(klasifikasi.getFiles(path), klasifikasi.getCategories(path))

old_class = klasifikasi.getCategories(path_data_uji)
new_class = klasifikasi.test(klasifikasi.getFiles(path_data_uji))
print(klasifikasi.hitungAkurasi(new_class, old_class))

print("End = ", datetime.datetime.now())
