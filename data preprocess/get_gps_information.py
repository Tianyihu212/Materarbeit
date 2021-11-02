#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:39:48 2021

@author: mac
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import re
import reverse_geocoder 
from bs4 import BeautifulSoup


def get_html(url): #定义函数
    headers = {
        'User-Agent':'Mozilla/5.0(Macintosh; Intel Mac OS X 10_11_4)\
        AppleWebKit/537.36(KHTML, like Gecko) Chrome/52 .0.2743. 116 Safari/537.36'
 
    }     #模拟浏览器访问
    response = requests.get(url,headers = headers)       #请求访问网站
    html = response.text       #获取网页源码
    return html                #返回网页源码
df_category = pd.read_csv('category1.csv')

df = pd.read_csv('data_new.csv') #读文件
index_unique = np.unique(df.landmark_id)
index_unique = index_unique[:100]
id_coordinates = np.empty((0,2))
id_captions = []
for index in tqdm(index_unique):
    url=df_category.category[index]
    soup = BeautifulSoup(get_html(url), 'lxml')   #初始化BeautifulSoup库,并设置解析器
#     for tbody in soup.find_all(name='tbody'):  
#         for tr in tbody.find_all(name='tr'): 
#             if tr.th == None:
#                 continue
#             if tr.th.string!='Location':
#                 pass
#             else:
#                 print(tr.td.a.string)      #输出结果
    

        
        
        
    # for tbody in soup.find_all(name='tbody'):
    swith = True    
    for a in soup.find_all(name='a'): 
        try:
            coordinates = np.array((float(a.attrs['data-lat']), float(a.attrs['data-lon'])))
            # print(coordinates)
            # reverse_geocoder.search(coordinates)
            id_coordinates = np.concatenate((id_coordinates, coordinates[np.newaxis,:]),axis = 0)
            swith = False
            break
        except:
            continue
    if swith == True:
        id_coordinates = np.concatenate((id_coordinates, np.array([[np.nan,np.nan]])),axis = 0)

    for caption in soup.find_all(name='title'):  
        id_captions.append(caption.string[9:-20])
        # print (caption.string)
        
df_histo = pd.DataFrame({'landmark_id':index_unique.flatten(),'gps_lat':id_coordinates[:,0],'gps_lon':id_coordinates[:,1],'name':id_captions})
df_histo.to_csv("gps_information.csv",index=False, sep=',')        
