# -*- coding: utf-8 -*-

import numpy as np

def map_at_k(y_true, y_pred):
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.ndim == 2 and y_pred.ndim == 2
    
    k = y_pred.shape[1]
    is_correct_list = []
    for i in range(y_true.shape[1]):
        print(y_true[:,i])
        is_correct = y_true[:,i][:,np.newaxis] == y_pred
        is_correct_list.append(is_correct)
        
    
    
    is_correct_mat = np.logical_or.reduce(np.array(is_correct_list))
    print(is_correct_mat)
    cumsum_mat = np.apply_along_axis(np.cumsum, axis=1, arr=is_correct_mat)
    arange_mat = np.expand_dims(np.arange(1, k + 1), axis=0)

    return np.mean((cumsum_mat / arange_mat) * is_correct_mat)

a= np.array([[1,np.nan],[4,5]])
b= np.array([[3,4,6,3,1],[4,5,4,5,4]])
print(map_at_k(a, b))
