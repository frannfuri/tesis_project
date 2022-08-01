import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import MAD, robust_z_score_norm
import scipy.stats
import os
from global_features_upgraded import get_cmap
import math

if __name__ == '__main__':
    f_analysis = 'IAF_post_chns'
    subj_name = 'SA007'
    path_days = '/home/jack/cluster_results/whole_record_results/'
    discrim_f = 'deltaPANSS_posit'
    directory_targets = '/home/jack/cluster_results/labels'

############################################
    # Load values of y
    discrim_f = 'deltaPANSS_posit'
    target_info = pd.read_csv('{}/{}_labels.csv'.format(directory_targets, subj_name), index_col=0, decimal=',')
    target_info = target_info.to_dict()
    target_info = target_info[discrim_f]
    df_rows = []
    for root, _, files in os.walk(path_days):
        for f in sorted(files):
            if subj_name in f:
                df_day = pd.read_csv(os.path.join(root, f), index_col=0)
                df_day['target'] = target_info[subj_name + '_' + f[-9:-4]]
                df_rows.append(df_day)
    df_all_days = pd.concat(df_rows)
    for v in df_all_days['SUBJ']:   # sanity check
        assert v == subj_name
    df_all_days = df_all_days.drop(columns='SUBJ')
    #df_all_days = df_all_days.sort_values(by=['target'])
    df_all_days = df_all_days.sort_values(by=[f_analysis])
    plt.figure(figsize=(13, 5))
    plt.plot(df_all_days[f_analysis], df_all_days['target'], '*-')
    plt.title('{}, f_analysis: {}\ntarget: {}'.format(subj_name, f_analysis, discrim_f))
    plt.grid()
    plt.tight_layout()
    plt.show()

