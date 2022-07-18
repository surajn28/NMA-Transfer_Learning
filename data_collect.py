import os
import h5py
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

character_list = ['aia', 'bonnie', 'jules', 'malcolm', 'mery', 'ray']
expression_list = ['joy', 'sadness']

labels_map = {
    'joy':0,
    'sadness':1
}

def generate_300(folder):

    for character in character_list :

        per_dir = os.path.join(folder,character)

        for expression in expression_list:

            per_exp_path = os.path.join(per_dir,f'{character}_{expression}')
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


def split_train_test(path):

    train_file = h5py.File('train.h5', 'w')
    test_file = h5py.File('test.h5', 'w')

    for expression in expression_list:
        
        train_mat = np.empty((1200, 128, 128, 3), dtype=np.uint8)
        test_mat = np.empty((600, 128, 128, 3), dtype=np.uint8)

        counter = 0

        for character in character_list :
        
            h5_path = os.path.join(path,f'{character}_{expression}.h5')
            h5 = h5py.File(h5_path,'r')
            i_pic_mat = h5['pic_mat']

            train_mat[counter*200:(counter+1)*200, :, :, :] = i_pic_mat[0:200, :, :, :]
            test_mat[counter*100:(counter+1)*100, :, :, :] = i_pic_mat[200:300, :, :, :]

            counter += 1

    train_file.create_dataset(expression,data=train_mat)
    test_file.create_dataset(expression,data=test_mat)
    
    train_file.close()
    test_file.close()
    print(train_mat.shape)


if __name__ =='__main__':

    folder = '/Users/gaojun/Documents/p1/NMA/FERG_DB_256'
    split_train_test(folder)
