"""
author: 
Suraj Neelakantan (suraj.neelakantan@oru.se)

Script to convert jpg file to a h5py dataset
"""

import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import h5py

size_x = 762
size_y = 562

new_size_x = 128
new_size_y = 128

h5_data_happy = np.empty((140, new_size_x, new_size_x, 3), dtype=np.uint8)

# Read jpg files and resize
for i in range(len(glob.glob(r'C:\Users\Suraj\Desktop\Happy\*.jpg'))):
    i_filename = glob.glob(r'C:\Users\Suraj\Desktop\Happy\*.jpg')[i]
    n = cv2.imread(i_filename)
    img_happy = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    res_happy = cv2.resize(img_happy, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(res_happy)
    # plt.show()
    # pdb.set_trace()
    h5_data_happy[i, :, :, :] = res_happy


# Converting all "happy" jpg images into a h5 dataset
happy_h5 = fr'KDEF_happy.h5'
f = h5py.File(happy_h5, 'w')
f.create_dataset('h5_data', data=h5_data_happy)
f.close()

h5_data_sad = np.empty((140, new_size_x, new_size_x, 3), dtype=np.uint8)

# Read jpg files and resize
for j in range(len(glob.glob(r'C:\Users\Suraj\Desktop\Sad\*.jpg'))):
    j_filename = glob.glob(r'C:\Users\Suraj\Desktop\Sad\*.jpg')[i]
    m = cv2.imread(j_filename)
    img_sad = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    res_sad = cv2.resize(img_sad, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(res_sad)
    # plt.show()
    h5_data_sad[i, :, :, :] = res_sad


# Converting all "sad" jpg images into a h5 dataset
sad_h5 = fr'KDEF_sad.h5'
h = h5py.File(sad_h5, 'w')
h.create_dataset('h5_data', data=h5_data_sad)
h.close()