#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tinayi Hu
Extract EXIF information of the image,which includes torage location,
shooting time, date, shooting focal length, camera exposure and other information.
"""
from PIL import Image
from PIL.ExifTags import TAGS 
import pandas as pd
import numpy as np
import glob, os
from tqdm import tqdm

def get_exif_data(fname):
    """
    Get embedded EXIF data from image file.

    Parameters
    ----------
    fname : string
        image address
    Returns
    -------
    ret : dictionary
        EXIF information of the image

    """
    ret = {}
    try:
        img = Image.open(fname)
        if hasattr( img, '_getexif' ):
            exifinfo = img._getexif()
            if exifinfo != None:
                for tag, value in exifinfo.items():
                    decoded = TAGS.get(tag, tag)
                    ret[decoded] = value
    except IOError:
        print ('IOERROR ' + fname)
    return ret



def get_time_in_day(time):
    """

    Parameters
    ----------
    time : int
        shooting time of image (hour)

    Returns
    -------
    a : string
        image shooting time period
    """

    a=' '
    if 6<=time<11:
        a='Morning'
    elif 11<=time<14:
        a='Noon'
    elif 14<=time<19:
        a='Afternoon'
    elif a<6 or a>=19:
        a='Night'
    return a    


def get_time_in_season(date):
    """

    Parameters
    ----------
    date : string
        Month of shooting image

    Returns
    -------
        b : int
        The month the image was shoot

    """
    b=' '
    if 3<= date <=5:
        b='Spring'
    elif 6<= date <=8:
        b='Summer'
    elif 9<=date <=11:
        b='Autumn'
    elif date==12 or date<=2:
        b='Winter'
    return b


if __name__ == '__main__':

    # load columns of the csv file include [landmark_id, image uri]
    # A row of the csv file represents an image
    df = pd.read_csv('data_new.csv')   
    index_unique = np.unique(df.landmark_id)
    path_new = 'training_data_clean_compressed/'
    index_unique = index_unique[:10]
    year = []
    time = []
    season = []
    boolean = []  # True if time and year information is successfully extracted
    time_period = []
    id_1 = []  # landmark_id
    for index in tqdm(index_unique):
        path="traindata/"+str(index)+"/*.*"
        images=glob.glob(path, recursive=True)

        #  Iterate through all images in GLD-v2 dataset
        for image in images:
            switch = False
            filename = os.path.basename(image)
            filetype = os.path.splitext(filename)[-1][1:]

            #  Filter out some unavailable image formats
            if filetype == 'tif' or filetype == 'TIF' or filetype == 'tiff' or filetype == 'TIFF' or filetype == 'gif' or filetype == 'GIF':
                continue

            #  extract information
            exif = get_exif_data(image)

            #  try to extract information else leave empty
            try:
                a = exif['DateTime'].split(' ')
                year.append(a[0])
                time.append(a[1])
                season.append(get_time_in_season(a[0].split(':')[0]))
                time_period.append(get_time_in_day(a[1].split(':')[0]))
                switch = True
                
            except:
                pass
            if switch == False:
                year.append(' ')
                time.append(' ')
            boolean.append(switch)
            id_1.append(filename)

    #  save csv file
    df_histo = pd.DataFrame({'landmark_id':id_1,'year':year,'time':time})
    df_histo.to_csv("exif_information.csv",index=False, sep=',')
