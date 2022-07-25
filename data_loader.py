from encodings import utf_8
import os
from tqdm import tqdm
import pandas as pd
os.chdir('/home/gaojud96/DL_model/Transfer_Learning')
from tokenize import Double
import numpy as np
import os.path as osp
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torchvision.io import read_image
import h5py
from PIL import Image
import torch 
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
#from torchvision.io import read_image
# import matplotlib.pyplot as plt
import csv
batch_size = 12

class HDF5Dataset(Dataset):

  def __init__(self, h5_path, transform=None):

    self.path = h5_path
    self.data = h5py.File(self.path,'r')
    self.labels_map = { 'joy':0,
                        'sadness':1
                        }

    self.length = 0
    for key in self.data.keys():
        self.length += len(self.data[key])
    
    self.images = np.empty((self.length,128,128,3),dtype=np.double) 
    self.labels = np.empty((self.length))

    counter = 0
    for key in self.data.keys():
        self.images[counter*len(self.data[key]):(counter+1)*len(self.data[key]),] = self.data[key]
        self.labels[counter*len(self.data[key]):(counter+1)*len(self.data[key])] = np.array([self.labels_map[key]]*len(self.data[key]))
        counter+=1
    
    self.labels = self.labels.astype(int)
    self.transform = transform


  def __getitem__(self, index):

    #img = Image.fromarray(self.images[index,],'RGB')
    img = self.images[index,]
    img = np.transpose(img, [2, 0, 1])
    img = torch.as_tensor(img)

    label = self.labels[index]
    
    if self.transform is not None:
      img = self.transform(img)
    return (img,label)
  
  def __len__(self):
    return self.length

class KDEFDataset(Dataset):
    # dataloader output: (pic_indices, color_channel, height, width)
    def __init__(self, h5_filename, transform=None, target_transform=None):
        self.h5_filename = h5_filename
        self.img_h5_file = self._load_h5_file(self.h5_filename)
        self.all_labels = self.img_h5_file['labels'][:]
        self.transform = transform  # it just assign the "state" of transform, not apply it


    def __len__(self):
        return len(self.all_labels)


    def __getitem__(self, idx):
        img = self.img_h5_file['img_data'][idx]
        label = self.img_h5_file['labels'][idx]

        label = torch.as_tensor(label, dtype=torch.float64)

        img = np.transpose(img, [2, 0, 1])
        img = img.astype(np.double)
        img = torch.as_tensor(img, dtype=torch.float64)
        # img = img/255
        if self.transform is not None:
            img = self.transform(img)  # we're going to write specific methods for transform
        return img, label

    def _load_h5_file(self, h5_filename):
        file = h5py.File(h5_filename, 'r')
        img_data = file['pic_mat']
        img_labels = file['labels']
        return dict(file=file, img_data=img_data, labels=img_labels)

class RAFDataset(Dataset):

  def __init__(self,imgdir,labeldir, transform=None):
    self.imgdir = imgdir
    self.labeldir = labeldir
    self.transform = transform
    label_map = {
      'Surprise':0,
      'Fear':1,
      'Disgust':2,
      'Happiness':3,
      'Sadness':4,
      'Anger':5,
      'Neutral':6
    }
    #self.imgs = [osp.join(self.imgdir,f) for f in os.listdir(self.imgdir)]
    self.data = pd.read_csv(self.labeldir)


  def __getitem__(self, index):
    
    img = read_image(osp.join(self.imgdir,self.data.iloc[index]['image']))
    img = img.double()
    if self.transform is not None:
      img = self.transform(img)
    return img,int(self.data.iloc[index]['label'])-1

  def __len__(self):
    return self.data.shape[0]



if __name__ == "__main__":

    # path = '/Users/gaojun/Documents/p1/NMA/FERG_DB_256'
    # trainset = HDF5Dataset(osp.join(path,'train.h5'))
    # testset = HDF5Dataset(osp.join(path,'test.h5'))
    # trainloader = DataLoader(trainset, batch_size=12, shuffle=True)
    # testloader = DataLoader(testset, batch_size=12, shuffle=True)

    imgdir = '/home/gaojud96/DL_model/Transfer_Learning/dataset/RAF-DB/basic/Image/aligned_224'
    train_labeldir = '/home/gaojud96/DL_model/Transfer_Learning/dataset/RAF-DB/train.csv'
    test_labeldir = '/home/gaojud96/DL_model/Transfer_Learning/dataset/RAF-DB/test.csv'
    trainset = RAFDataset(imgdir,train_labeldir)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = RAFDataset(imgdir,train_labeldir)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

