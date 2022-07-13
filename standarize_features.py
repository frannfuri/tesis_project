import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import robust_z_score_norm, minmax_norm

csv_path = 'results_bef_13-07/whole_selected_features_data_60sec.csv'
subjs_names = ['SA007', 'SA010', 'SA014', 'SA017', 'SA047']


df = pd.read_csv(csv_path, index_col=0)
all_df_subjs = []
for subj in subjs_names:
    print('Standarization of {}'.format(subj))
    df_subj = df[df['SUBJ']==subj]
    df_subj['PANSS'] = minmax_norm(df_subj['PANSS'])
    df_subj['PANSS_posit'] = minmax_norm(df_subj['PANSS_posit'])
    column_names = list(df_subj.keys())
    for feat in column_names:
        if feat != 'SUBJ' and feat != 'PANSS' and feat!= 'PANSS_posit':
            df_subj[feat] = robust_z_score_norm(df_subj[feat])
    all_df_subjs.append(df_subj)
standarized_df = pd.concat(all_df_subjs, axis=0)
standarized_df.to_csv(csv_path[:-4] + '_standard.csv')