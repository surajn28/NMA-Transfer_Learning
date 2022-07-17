import os
import h5py
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


folder = '/Users/gaojun/Documents/p1/NMA/FERG_DB_256'
for person in os.listdir(folder):

    if person.endswith('.txt'):continue

    per_dir = os.path.join(folder,person)

    for expression in os.listdir(per_dir):

        per_exp_path = os.path.join(per_dir,expression)
        pics = per_exp_path+'/*.png'

        pic_filenames = glob.glob(pics)
        shuffle = list(np.random.permutation(len(pic_filenames))[0:300])

        pic_choose = []

        for i in shuffle:
            pic_choose.append(pic_filenames[i])

        pic_mat = np.empty((300, 128, 128, 3), dtype=np.uint8)

        # (300, 25, 72, 3)

        for idx_file in range(len(pic_choose)):
            i_filename = pic_choose[idx_file]
            i_pic = cv2.imread(i_filename)
            if i_pic is None: continue
            i_pic_correct = cv2.cvtColor(i_pic, cv2.COLOR_BGR2RGB)
            i_pic_resized = cv2.resize(i_pic_correct, dsize=(128, 128))
            pic_mat[idx_file, :, :, :] = i_pic_resized



        f_name = f'{expression}.h5'
        f = h5py.File(f_name, 'w')
        f.create_dataset('pic_mat', data=pic_mat)
        f.close()
