from Klasifikasi1 import Klasifikasi
import os
import glob
import errno
import datetime

print("Start = ", datetime.datetime.now())
path = input('Input Direktori: ')
path_data_uji = input("Input Data Uji: ")
# files_in_path = glob.glob(path)

# files = []
# files_cat = []

# for file in files_in_path:
#     try:
#         with open(file) as f:
#             x = f.read()
#             files.append(x)
#     except IOError as exc:
#         if exc.errno != errno.EISDIR:
#             raise

#     files_cat.append(os.path.split(os.path.dirname(file))[-1])


klasifikasi = Klasifikasi()

klasifikasi.train(klasifikasi.getFiles(path), klasifikasi.getCategories(path))

old_class = klasifikasi.getCategories(path_data_uji)
new_class = klasifikasi.test(klasifikasi.getFiles(path_data_uji))
print(klasifikasi.hitungAkurasi(new_class, old_class))

print("End = ", datetime.datetime.now())
