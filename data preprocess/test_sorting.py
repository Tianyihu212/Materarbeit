# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 17:57:00 2021

@author: M S I
"""
import shutil
import pandas as pd
import numpy as np
import glob, os
from PIL import Image
from tqdm import tqdm

def mkdir(path):
    """
    create a new folder
    Parameters
    ----------
    path : str
        address of folder
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

df = pd.read_csv('test_image.csv')
image_list = df.id.to_list()
landmarks_list = df.landmarks.to_list()
Usage_list = df.Usage.to_list()
path = 'H:\\test_image\\'
path_new = 'H:\\test_image_sort\\'
folder_list_1 = os.listdir(path)[2:]

for i in folder_list_1:
    path_1 = os.path.join(path, i)
    
    folder_list_2 = os.listdir(path_1)
    for j in folder_list_2:
        path_2 = os.path.join(path_1,j)
        
        folder_list_3 = os.listdir(path_2)
        for k in tqdm(folder_list_3):
            path_3 = os.path.join(path_2,k)
            
            folder_list_4 = os.listdir(path_3)
            for l in folder_list_4:
                path_4 = os.path.join(path_3,l)
                
                images = glob.glob(path_4+"\\*.*")
                
                for image in images:
                    basename = os.path.basename(image).split('.')[0]
                    index = image_list.index(basename)
                    if Usage_list[index]=="Public":
                        tmp_path = os.path.join(path_new,'public')
                    else:
                        tmp_path = os.path.join(path_new,'private')
                    if isinstance(landmarks_list[index],float):
                        tmp_path = os.path.join(tmp_path, 'nolabel')
                        shutil.copy(image, tmp_path)
                    else:
                        tmp_path = os.path.join(tmp_path, 'label')
                        shutil.copy(image, tmp_path)
                    # else:
                    #     ids = landmarks_list[index].split()
                    #     for m in ids:
                    #         tmp_path_ = os.path.join(tmp_path, m)
                    #         mkdir(tmp_path_)
                    #         shutil.copy(image, tmp_path_)
                            
                            
                            
                        
                   
                    