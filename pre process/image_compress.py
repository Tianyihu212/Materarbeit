# -*- coding: utf-8 -*-
"""
Created on Tianyi Hu
"""
import pandas as pd
import numpy as np
import glob, os
from PIL import Image
from tqdm import tqdm
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                 
		os.makedirs(path) 
        
df = pd.read_csv('data_new.csv')   
index_unique = np.unique(df.landmark_id)
path_new = 'training_data_clean_compressed/'
index_unique = index_unique[11:13]

for index in tqdm(index_unique):
    path="train_dataset\\"+str(index)+"/*.*"
    images=glob.glob(path, recursive=True)
    
    mkdir(path_new+str(index))
    
    for image in images:
        filename = os.path.basename(image)
        filetype = os.path.splitext(filename)[-1][1:] 
        if filetype == 'tif' or filetype == 'TIF' or filetype == 'tiff' or filetype == 'TIFF' or filetype == 'gif' or filetype == 'GIF':
            continue 
        try:
            with open(image.lower(), 'rb') as f:
                img = Image.open(f)
                new_image = img.resize((224, 224))  
                # new_image.save(path_new+str(index)+'/'+filename)
        except Exception as e:
            print(str(e))
            continue

    

 
