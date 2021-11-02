#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tianyi Hu
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import re
import reverse_geocoder 
from bs4 import BeautifulSoup

if __name__ == '__main__':
    # freeze_support()
    df = pd.read_csv('gps_information.csv')
    
    gps_lat = df.gps_lat
    gps_lon = df.gps_lon
    boolean = df.found
    index_unique = []
    countries = []
    for i in tqdm (range(len(gps_lat))):
        index_unique.append(df.landmark_id[i])
        if boolean[i] == False:
            countries.append(' ')
            continue
        coordinate = (gps_lat[i],gps_lon[i])
        country = reverse_geocoder.search(coordinate)[0]['cc']
        countries.append(country)

    df_histo = pd.DataFrame({'landmark_id':index_unique,'countries':countries})
    df_histo.to_csv("country_information.csv",index=False, sep=',')        


# coordinates = (-37.81, 144.96), (31.76, 35.21)

