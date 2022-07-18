# python script
# -*- coding: utf-8
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
I'd like to thank https://github.com/namiyousef/multi-task-learning/blob/main/data/data.py
for their amazing tutorial
"""
import os
import h5py
import pandas as pd
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from config.config import work_dir
import torchvision.transforms as transforms
# import torchvision.transforms

class H5FileDataset(Dataset):
    # dataloader output: (pic_indices, color_channel, height, width)

    def __init__(self, h5_filename, transform=None, target_transform=None):
        self.h5_filename = h5_filename
        self.img_h5_file = self._load_h5_file(self.h5_filename)
        self.all_labels = self.img_h5_file['labels'][:]


    def __len__(self):
        return len(self.all_labels)


    def __getitem__(self, idx):
        img = self.img_h5_file['img_data'][idx]
        label = self.img_h5_file['labels'][idx]
        # sample = {'data': img,
        #           'label': label,
        #           'img_idx': idx}
        # img = img.transform([2, 0, 1])
        img = np.transpose(img, [2, 0, 1])
        img = img.astype(np.double)
        return img, label


    def _load_h5_file(self, h5_filename):
        file = h5py.File(h5_filename, 'r')
        img_data = file['pic_mat']
        img_labels = file['labels']
        # img_data.transpose((2, 0, 1))
        return dict(file=file, img_data=img_data, labels=img_labels)


# train_dataset = H5FileDataset(r'D:\wliwa\Desktop\FERG_DB_256\h5_files\train_dataset.h5')
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
# batch_images, batch_labels = next(iter(train_dataloader))
# print('Batch size:', batch_images.shape)
# plt.figure()
# plt.imshow(batch_images[0].permute([1, 2, 0]).to(torch.uint8))
# plt.show()
#
#
# test_dataset = H5FileDataset(r'D:\wliwa\Desktop\FERG_DB_256\h5_files\test_dataset.h5')
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True)
# batch_images, batch_labels = next(iter(test_dataloader))
# print('Batch size:', batch_images.shape)
# plt.figure()
# plt.imshow(batch_images[0].permute([1, 2, 0]).to(torch.uint8))
# plt.show()
#

kdef_test_dataset = H5FileDataset(r'D:\wliwa\Desktop\KDEF\h5_files\kdef_test_dataset.h5')
test_dataloader = torch.utils.data.DataLoader(kdef_test_dataset, batch_size=50, shuffle=True)
batch_images, batch_labels = next(iter(test_dataloader))
print('Batch size:', batch_images.shape)
plt.figure()
plt.imshow(batch_images[0].to(torch.uint8).permute([1, 2, 0]).to(torch.uint8))
plt.show()


# transform_list = [transforms.RandomRotation(10)]
transform_list = [transforms.RandomRotation(90), transforms.CenterCrop([562, 562])]
aug = transforms.Compose(transform_list)

# aug = transforms.CenterCrop([64, 64])
for x, y in test_dataloader:
    # x = x.permute([0, 3, 1, 2])  # x: (idx, color_channel, height, width)
    x_aug = aug(x)
plt.imshow(x_aug[0].permute([1, 2, 0]).to(torch.uint8))




