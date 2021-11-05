# -*- coding: utf-8 -*-
"""
Created on Tianyi Hu
This file is used to compress image to certain size
"""
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

path_old = "H:\\test_image_sort\\public\\nolabel\\*.*"

# path to store compress image
path_new = "H:\\test_image_compress\\public\\nolabel\\"

images = glob.glob(path_old)


# Iterate all images in current landmark id
for image in tqdm(images):
    filename = os.path.basename(image)
    filetype = os.path.splitext(filename)[-1][1:]

    # Filter out all images that do not conform to the format
    if filetype == 'tif' or filetype == 'TIF' or filetype == 'tiff' or filetype == 'TIFF' or filetype == 'gif' or filetype == 'GIF':
        continue 

    # compress image and save image
    try:
        with open(image.lower(), 'rb') as f:
            img = Image.open(f)

            # compress image to 224*224 pixel
            new_image = img.resize((224, 224))  
            new_image.save(path_new+filename)
    except Exception as e:
        print(str(e))
        continue