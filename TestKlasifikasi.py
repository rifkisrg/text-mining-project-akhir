from Klasifikasi import Klasifikasi
import os
import glob
import errno
import datetime

#training
print("Start Train = ", datetime.datetime.now())

path = './Data/Data coba/*/*.txt'
files_in_path = glob.glob(path)

files = []
files_cat = []

for file in files_in_path:
    try:
        with open(file) as f:
            x = f.read()
            files.append(x)
    except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise

    files_cat.append(os.path.split(os.path.dirname(file))[-1])

klasifikasi = Klasifikasi()

klasifikasi.train(files, files_cat)
print("End Train = ", datetime.datetime.now())

#uji
print("Start Uji = ", datetime.datetime.now())

path = './Data/Data coba uji/*/*.txt'
files_in_path = glob.glob(path)

files = []
files_cat = []

for file in files_in_path:
    try:
        with open(file) as f:
            x = f.read()
            files.append(x)
    except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise

    files_cat.append(os.path.split(os.path.dirname(file))[-1])

klasifikasi = Klasifikasi()

arr_class_result = klasifikasi.test(files)

print("Akurasi sebesar = ", klasifikasi.hitungAkurasi(arr_class_result, files_cat))
print("End Train = ", datetime.datetime.now())
