# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com

Codes:
Example: AF01ANFL.JPG
Letter 1: Session
A = series one
B = series two

Letter 2: Gender
F = female
M = male

Letter 3 & 4: Identity number
01 - 35

Letter 5 & 6: Expression
AF = afraid
AN = angry
DI = disgusted
HA = happy
NE = neutral
SA = sad
SU = surprised

Letter 7 & 8: Angle
FL = full left profile
HL = half left profile
S = straight
HR = half right profile
FR = full right profile
Extension: Picture format
JPG = jpeg (Joint Photographic Experts Group)

"""
import os
import h5py
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

work_dir = r'D:\wliwa\Desktop\KDEF'
os.chdir(work_dir)

character_list = ['01', '02', '03', '04', '05', '06', '07']
character_list_female = [f'F{i_subject:02d}' for i_subject in range(1, 36)]
character_list_male = [f'M{i_subject:02d}' for i_subject in range(1, 36)]
character_list = character_list_female + character_list_male


# expression_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
expression_list = ['joy', 'sadness']
expression_code = {'joy': 'HA', 'sadness': 'SA'}

# saved_pic_size = 128
# n_pic_choose = 300
# n_pic_train = 200
# n_pic_test = 100
size_row = 762
size_col = 562


df_train_indices = pd.DataFrame(columns=['character', 'expression', 'idx'])
df_test_indices = pd.DataFrame(columns=['character', 'expression', 'idx'])

counter_train = 0
for i_character in character_list:
    for i_expression in expression_list:
        print(f'here is {i_character}, {i_expression}')
        i_train_indices = pd.DataFrame(np.zeros([1, 3], dtype=object), columns=['character', 'expression', 'idx'])
        i_test_indices = pd.DataFrame(np.zeros([1, 3], dtype=object), columns=['character', 'expression', 'idx'])

        i_expression_code = expression_code[i_expression]
        if len(glob.glob(fr'{i_expression}\*{i_character}{i_expression_code}S.JPG'))>0:
            i_train_indices['idx'] = i_character[1:]
            i_train_indices['character'] = i_character
            i_train_indices['expression'] = i_expression

            i_test_indices['idx'][0] = i_character[1:]
            i_test_indices['character'] = i_character
            i_test_indices['expression'] = i_expression

            df_train_indices = pd.concat([df_train_indices, i_train_indices])
            df_test_indices = pd.concat([df_test_indices, i_test_indices])

df_train_indices = df_train_indices.reset_index()
df_test_indices = df_test_indices.reset_index()

df_train_indices.to_csv('train_indices.csv')
df_test_indices.to_csv('test_indices.csv')


# dataset_types = ['train', 'test']
dataset_types = ['test']

for i_type in dataset_types:
    df_indices = pd.read_csv(f'{i_type}_indices.csv')
    n_pics = len(df_indices)
    pic_mat = np.empty((n_pics, size_row, size_col, 3), dtype=np.uint8)
    labels = np.empty(n_pics, dtype=object)
    labels_identity = np.empty(n_pics, dtype=object)
    pic_indices = np.empty(n_pics)

    for i_row in range(n_pics):
        i_character = df_indices['character'][i_row]
        i_expression = df_indices['expression'][i_row]
        i_expression_code = expression_code[i_expression]
        pic_idx = df_indices['idx'][i_row]
        i_filename = glob.glob(fr'{i_expression}\*{i_character}{i_expression_code}S.JPG')[0]
        i_pic = cv2.imread(i_filename)

        i_pic_correct = cv2.cvtColor(i_pic, cv2.COLOR_BGR2RGB)
        pic_mat[i_row, :, :, :] = i_pic_correct
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


test_name = fr'h5_files\test_dataset.h5'
test_set = h5py.File(test_name, 'r')['pic_mat']
test_set_labels = h5py.File(test_name, 'r')['labels'][:]
test_set_labels.astype(np.compat.unicode)
plt.imshow(test_set[-1, ])
