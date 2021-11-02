# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:49:25 2021

@author: Joachim
"""

import pandas as pd
import numpy as np




df=pd.read_csv('train.csv',sep=',')



# name = range(1,10000)
# df_clean = pd.read_csv('train_clean.csv',names=name,sep=' |,',index_col=0)
# name = range(1,100)
# df_clean = pd.read_csv('train_clean.csv',names=name,sep=' |,',index_col=0)

df_clean = pd.read_csv('train_clean.csv',sep=',',index_col=0)
df_index = pd.read_csv('train_clean.csv',usecols=[0])
df_index = np.array(df_index)
df_new=pd.DataFrame(columns=('id','url','landmark_id'))

for l_id in df_index:
    l_id = int(l_id)
    a = df_clean.loc[l_id].str.split(' ',expand=True)
    df_ = df[df.landmark_id==l_id]
    pic=df_[np.in1d(df_.id, a)]
    df_new=pd.concat([df_new,pic], axis=0)
# pic=df_[np.in1d(df_.id, a)] 代替了for循环
    print(l_id)
    # l_id = landmarkid
        
        
        





