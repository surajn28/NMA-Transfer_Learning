# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com

this scripts select 300 images from each condition in FERG database, 
resize (256, 256, 3) png format images to (128, 128, 3), 
then stack them to form an (300, 128, 128, 3) array, 
and saves the array to h5 files

"""
import os
import h5py
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

work_dir = r'D:\wliwa\Desktop\FERG_DB_256\FERG_DB_256'
os.chdir(work_dir)

character_list = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
# expression_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
expression_list = ['joy', 'sadness']

saved_pic_size = 128
n_pic_choose = 300

for i_character in character_list:
    for i_expression in expression_list:
        print(f'here is {i_character}, {i_expression}')
        cond_folder = fr'{i_character}\{i_character}_{i_expression}'
        n_pics_per_cond = len(glob.glob(cond_folder + r'\*.png'))
        shuffle_indices = np.sort(np.random.permutation(n_pics_per_cond)[0:n_pic_choose] + 1)
        pic_mat = np.empty((n_pic_choose, saved_pic_size, saved_pic_size, 3), dtype=np.uint8)

        for idx_file in range(n_pic_choose):
            pic_idx = shuffle_indices[idx_file]
            i_filename = fr'{i_character}\{i_character}_{i_expression}\{i_character}_{i_expression}_{pic_idx}.png'
            i_pic = cv2.imread(i_filename)

            i_pic_correct = cv2.cvtColor(i_pic, cv2.COLOR_BGR2RGB)
            i_pic_resized = cv2.resize(i_pic_correct, dsize=(saved_pic_size, saved_pic_size))
            pic_mat[idx_file, :, :, :] = i_pic_resized

        f_name = fr'h5_files\{i_character}_{i_expression}.h5'
        f = h5py.File(f_name, 'w')
        f.create_dataset('pic_mat', data=pic_mat)
        f.create_dataset('choose_indices', data=shuffle_indices)
        f.close()





