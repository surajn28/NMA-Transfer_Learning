# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
"""
import os
import h5py
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

work_dir = r'D:\wliwa\Desktop\FERG_DB_256'
os.chdir(work_dir)

character_list = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
# expression_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
expression_list = ['joy', 'sadness']

saved_pic_size = 128
n_pic_choose = 300
n_pic_train = 200
n_pic_test = 100


df_train_indices = pd.DataFrame(columns=['character', 'expression', 'idx'])
df_test_indices = pd.DataFrame(columns=['character', 'expression', 'idx'])

for i_character in character_list:
    for i_expression in expression_list:
        print(f'here is {i_character}, {i_expression}')
        i_train_indices = pd.DataFrame(columns=['character', 'expression', 'idx'])
        i_test_indices = pd.DataFrame(columns=['character', 'expression', 'idx'])

        cond_folder = fr'{i_character}\{i_character}_{i_expression}'
        n_pics_per_cond = len(glob.glob(cond_folder + r'\*.png'))
        shuffle_indices = np.random.permutation(n_pics_per_cond)[0:n_pic_choose] + 1
        train_indices = np.sort(shuffle_indices[0:200])
        test_indices = np.sort(shuffle_indices[200:])
        i_train_indices['idx'] = train_indices
        i_train_indices['character'] = i_character
        i_train_indices['expression'] = i_expression

        i_test_indices['idx'] = test_indices
        i_test_indices['character'] = i_character
        i_test_indices['expression'] = i_expression

        df_train_indices = pd.concat([df_train_indices, i_train_indices])
        df_test_indices = pd.concat([df_test_indices, i_test_indices])

df_train_indices = df_train_indices.reset_index()
df_test_indices = df_test_indices.reset_index()

df_train_indices.to_csv('train_indices.csv')
df_test_indices.to_csv('test_indices.csv')


dataset_types = ['train', 'test']

for i_type in dataset_types:
    df_indices = pd.read_csv(f'{i_type}_indices.csv')
    n_pics = len(df_indices)
    pic_mat = np.empty((n_pics, saved_pic_size, saved_pic_size, 3), dtype=np.uint8)
    labels = np.empty(n_pics, dtype=object)
    labels_identity = np.empty(n_pics, dtype=object)
    pic_indices = np.empty(n_pics)

    for i_row in range(n_pics):
        i_character = df_indices['character'][i_row]
        i_expression = df_indices['expression'][i_row]
        pic_idx = df_indices['idx'][i_row]
        i_filename = fr'{i_character}\{i_character}_{i_expression}\{i_character}_{i_expression}_{pic_idx}.png'
        i_pic = cv2.imread(i_filename)

        i_pic_correct = cv2.cvtColor(i_pic, cv2.COLOR_BGR2RGB)
        i_pic_resized = cv2.resize(i_pic_correct, dsize=(saved_pic_size, saved_pic_size))
        pic_mat[i_row, :, :, :] = i_pic_resized
        labels[i_row] = i_expression
        labels_identity[i_row] = i_character
        pic_indices[i_row] = pic_idx

    f_name = fr'h5_files\{i_type}_dataset.h5'
    f = h5py.File(f_name, 'w')
    f.create_dataset('pic_mat', data=pic_mat)
    f.create_dataset('choose_indices', data=pic_indices)
    f.create_dataset('labels', data=labels)
    f.create_dataset('labels_identity', data=labels_identity)
    f.close()


# train_name = fr'h5_files\train_dataset.h5'
# train_set = h5py.File(train_name, 'r')['pic_mat'][0:1200, ]
#
# test_name = fr'h5_files\test_dataset.h5'
# test_set = h5py.File(test_name, 'r')['pic_mat']

