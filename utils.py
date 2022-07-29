import numpy as np
import pandas as pd

def mean_power(signal):
    signal_sum = np.sum(signal)
    return abs(signal_sum)/len(signal)

def MAD(data):
    M = np.median(data)
    diff_vector = []
    for x in data:
        diff_vector.append(np.abs(x-M))
    return np.median(np.array(diff_vector))

def robust_z_score_norm(data):
    norm_data = []
    MAD_data = MAD(data)
    for x in data:
        num_x = 0.6745*(x-np.median(data))
        norm_x = num_x/MAD_data
        norm_data.append(norm_x)
    return np.array(norm_data)

def minmax_norm(data):
    maxdata = max(data)
    mindata = min(data)
    norm_data = []
    if maxdata == mindata:
        for i in data:
            norm_data.append(0.5)
    else:
        for i in data:
            norm_val = (i - mindata)/(maxdata - mindata)
            norm_data.append(norm_val)
    return np.array(norm_data)

def value(x):
    if isinstance(x, tuple):
        if len(x)>1:
            raise AssertionError('Error! Value is a tuple.')
        x = x[0]
    return x