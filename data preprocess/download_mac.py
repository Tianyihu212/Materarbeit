# -*- coding: utf-8 -*-
"""
Created on Tianyi Hu
In this file i use code to download GLD-v2 dataset.
Here I use the uri link provided by the GLD-v2 official website to download.
The uri link file can be found at the following address:
https://github.com/cvdfoundation/google-landmark to download.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests


def mkdir(path):
    """
    create new folder under the given path
    Parameters
    ----------
    path : string
    Path to create folder

    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


# Defines the maximum number of pictures downloaded for each landmark_id.
max_pic =50

# load csv file
# Each row of csv corresponds to an image.
# The column of csv includes [index, landmark_id, image uri]
df = pd.read_csv('data_new.csv')

# Simulate computer behavior with python
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)\
            AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}

file = "training_data_clean/"
count = 1
current_landmark = 0

# Iterate over the read csv file.
for idx, row in tqdm(df.iterrows()):
    landmark_id = row['landmark_id']

    # Check whether the current download number exceeds the set maximum download value.
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

    # GLD-v2 data save address
    path = file+str(landmark_id)
    mkdir(path)

    # try to download image
    try:
        resp = requests.get(url,headers=headers)
        file_name = path+'/'+idxx + '.' + url.split('.')[-1]
        
        with open(file_name, 'wb') as f:
            f.write(resp.content)
    except:
        pass
