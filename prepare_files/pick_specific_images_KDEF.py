"""
author: 
Suraj Neelakantan (suraj.neelakantan@oru.se)

Script to pick only "happy" and "sad" emotion images from KDEF dataset
"""
import os
import shutil
from glob import glob
import pdb

rootdir = r'C:\Users\Suraj\Desktop\KDEF_and_AKDEF\KDEF'

happy_dest = r'C:\Users\Suraj\Desktop\Happy'
sad_dest = r'C:\Users\Suraj\Desktop\Sad'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:

        if file[-7:] == 'HAS.JPG':
            print('Happy faces - straight:', file)
            shutil.copy(subdir + "/" + file, happy_dest)
        
        if file[-7:] == 'SAS.JPG':
            print('Sad faces - straight:', file)
            shutil.copy(subdir + "/" + file, sad_dest)