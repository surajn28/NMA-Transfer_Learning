from encodings import utf_8
import os
import numpy as np
import os.path as osp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import h5py
from PIL import Image 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class HDF5Dataset(Dataset):

  def __init__(self, h5_path):

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
      
  def __getitem__(self, index):

    #img = Image.fromarray(self.images[index,],'RGB')
    img = self.images[index,]
    label = self.labels[index]

    #image = self.transform(img)
    return (img.transpose((2,0,1)),label)
  
  def __len__(self):
    return self.length

if __name__ == "__main__":

    path = '/Users/gaojun/Documents/p1/NMA/FERG_DB_256'
    trainset = HDF5Dataset(osp.join(path,'train.h5'))
    testset = HDF5Dataset(osp.join(path,'test.h5'))

    trainloader = DataLoader(trainset, batch_size=12, shuffle=True)
    testloader = DataLoader(testset, batch_size=12, shuffle=True)
