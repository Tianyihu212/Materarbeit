# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:34:16 2021

@author: Joachim
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                 
		os.makedirs(path) 


max_pic =20

df = pd.read_csv('data_new.csv')
df = df.head(800037)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
            AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}
df = df[94473:]

file = "training_data_clean/"
count = 1
current_landmark = 0

for idx, row in tqdm(df.iterrows()):
    landmark_id = row['landmark_id']
    
    if landmark_id == current_landmark:
        count += 1
        
        if count > max_pic:
            continue
        
    else:
        print(landmark_id)
        current_landmark = landmark_id
        count = 1

    url = row['url']
    idxx = row['id']

    path = file+str(landmark_id)
    mkdir(path)
    try:
        resp = requests.get(url,headers=headers)
        file_name = path+'/'+idxx + '.' + url.split('.')[-1]
        
        with open(file_name, 'wb') as f:
            f.write(resp.content)
    except:
        pass






