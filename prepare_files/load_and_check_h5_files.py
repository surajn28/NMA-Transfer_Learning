# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com

this scripts opens the h5 files to check if we stored it correctly, 
and plot 15 out of 300 images to check quality

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

for i_character in character_list:
    for i_expression in expression_list:
        print(f'here is {i_character}, {i_expression}')
        cond_folder = fr'{i_character}\{i_character}_{i_expression}'

        file = h5py.File(fr'h5_files\{i_character}_{i_expression}.h5', 'r')
        pic_mat = file['pic_mat'][:]
        choose_indices = file['choose_indices'][:]

        # for idx_pic in range(pic_mat.shape[0]):
        for idx_pic in range(0, 300, 20):
            print(idx_pic)
            plt.figure()
            plt.imshow(pic_mat[idx_pic, ])

        file.close()

