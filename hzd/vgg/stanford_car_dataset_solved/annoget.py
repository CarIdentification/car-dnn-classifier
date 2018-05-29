import scipy.io as sio
import numpy as np
import os, shutil

base_path = "train"
load_fn = 'cars_train_annos.mat'
load_data = sio.loadmat(load_fn)
annos = load_data['annotations']
fclass = load_data['annotations']['class'][0]
fnames = load_data['annotations']['fname'][0]

# 196类汽车
for i in range(1, 197):
    path = base_path + os.path.sep + str(i)
    if not os.path.exists(path):
        os.mkdir(path)

for i in range(fnames.size):
    src = base_path + os.path.sep + fnames[i][0]
    dst = base_path + os.path.sep + str(fclass[i][0][0]) + os.path.sep + fnames[i][0]
    print("%s %s \n" % (src, dst))
    shutil.move(src, dst)
