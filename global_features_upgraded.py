import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import MAD, robust_z_score_norm
import scipy.stats
import math

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
#['skew_Theta', 'kurt_Theta', 'skew_Delta', 'AlphaBand_low_bound', 'std_Alpha', 'kurt_Alpha']
if __name__=='__main__':
    f_analysis = 'LZC_slope_H-pow_Complete'
    subj_name = 'SA047'
    discrim_f = 'deltaPANSS_posit'
    # Window length dependents
    path_csv = '/home/jack/cluster_results/whole_features_data_30sec_fixed.csv'
    path_n_windows = '/home/jack/cluster_results/n_windows/whole_n_windows_data_30sec.txt'

    directory_targets = '/home/jack/cluster_results/labels'
    n_bins = 25 # for the histogram
##############################################
    df_features = pd.read_csv(path_csv, index_col=0)
    df_features = df_features.drop(columns=['PANSS', 'PANSS_posit'])  # remove target columns, the specified
                                                                # feature target will be added later
    # Add record_id info
    df_n_windows = pd.read_csv(path_n_windows, index_col=0, header=None)
    df_n_windows = df_n_windows.to_dict()
    df_n_windows = df_n_windows[1]
    new_col = []
    for key, value in df_n_windows.items():
        new_segment = value * [key]
        new_col = [*new_col, *new_segment]

    df_features['record_ID'] = new_col

    for i in range(df_features.shape[0]): # sanity check
        assert df_features['record_ID'][i][:5] == df_features['SUBJ'][i]

    df_features = df_features[df_features['SUBJ']==subj_name]
    df_features = df_features.drop(columns=['SUBJ']) # Now the SUBJ info is reduntant with record_ID info,
                                        # and probably we are using one subject at once

    for i in range(df_features.shape[0]): # sanity check
        assert df_features['record_ID'].iloc[i].startswith(subj_name)
    target_info = pd.read_csv('{}/{}_labels.csv'.format(directory_targets, subj_name), index_col=0, decimal=',')
    target_info = target_info.to_dict()
    target_info = target_info[discrim_f]
    target_col = []
    for r in range(df_features.shape[0]):
        target_col.append(target_info[df_features['record_ID'].iloc[r]])
    df_features['target'] = target_col
    df_features = df_features.dropna()

    list_of_records = df_features['record_ID'].unique()

    # Sort record ids by target value
    y_per_day = []
    for rec in list_of_records:
        y_per_day.append(np.median(df_features['target'][df_features['record_ID']==rec]))
        assert y_per_day[-1] == df_features['target'][df_features['record_ID']==rec].iloc[0]
        assert all(t_val == df_features[df_features['record_ID'] == rec].iloc[0]['target'] for t_val in
                   df_features[df_features['record_ID'] == rec]['target'])
    df2 = pd.DataFrame({'target': y_per_day, 'record_day': list_of_records})
    df2 = df2.sort_values(by=['target'])
    list_of_records = df2['record_day'].values
    vals, names, xs = [], [], []
    for i in range(len(list_of_records)):
        vals.append(df_features[f_analysis][df_features['record_ID']==list_of_records[i]].values)
        names.append('{}\ny={:.3f}'.format(list_of_records[i], df2['target'].iloc[i]))
        xs.append(np.random.normal(i + 1, 0.04,
                                   df_features[f_analysis][df_features['record_ID'] == list_of_records[i]].values.shape[0]))
    plt.figure(figsize=(13,5))
    plt.boxplot(vals, labels=names, vert=False)
    plt.grid()
    cmap = get_cmap(len(vals))
    colors_ = []
    for i in range(len(vals)):
        colors_.append(cmap(i))
    for x, val, c in zip(xs, vals, colors_):
        plt.scatter(val, x, alpha=0.4, color=c)
    plt.xlabel(f_analysis)
    plt.ylabel(r'$\Delta$ {}'.format(discrim_f[5:]))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-1, 2))
    plt.title('{} {} [w_len={}]\ntarget_feature: {}'.format(f_analysis, subj_name, path_csv[-16:-10], discrim_f))
    plt.tight_layout()
    plt.show()

    '''
    fig, axs = plt.subplots(int(np.ceil(len(list_of_records) / 3)), 3, figsize=(15, 15))
    ax_n = 0
    fig.suptitle(
        'Histograms per {}, of feature {}, {} window length {} seconds'.format(discrim_f, f_analysis, subj_name,
                                                                               path_csv[-16:-10]), fontsize=10)
    min_val = min(df_features[f_analysis])
    max_val = max(df_features[f_analysis])
    for i in range(len(list_of_records)):
        if np.ceil(len(list_of_records) / 3) < 2:
            axs[ax_n].hist(df_features[f_analysis][df_features['record_ID'] == list_of_records[i]], n_bins)
            axs[ax_n].set_xlim((min_val, max_val))
            axs[ax_n].ticklabel_format(axis='x', style='sci', scilimits=(-1,2))
            axs[ax_n].set_title('{} = {}'.format(discrim_f, df2['target'].iloc[i]), fontsize=8)
            ax_n += 1
        else:
            if i != 0 and i % 3 == 0:
                ax_n += 1
            axs[ax_n, i % 3].hist(df_features[f_analysis][df_features['record_ID'] == list_of_records[i]], n_bins)
            axs[ax_n, i % 3].set_xlim((min_val, max_val))
            axs[ax_n, i % 3].ticklabel_format(axis='x', style='sci', scilimits=(-1,2))
            axs[ax_n, i % 3].set_title('{}\n{} = {}'.format(list_of_records[i], discrim_f, df2['target'].iloc[i]), fontsize=8)
    plt.subplots_adjust(hspace=0.5)
    plt.show(block=False)
    a = 0
    df_features = df_features.drop(columns=['AlphaBand_low_bound', 'AlphaBand_high_bound', 'CoG_post_chns'])
    ids_to_remove =[]
    for rec in list_of_records:
        for col in range(df_features[df_features['record_ID']==rec].shape[1]):
            if col == df_features.shape[1]-1 or col == df_features.shape[1]-2:
                continue
           #normed_features = robust_z_score_norm(df_features[df_features['record_ID']==rec].iloc[:,col])
            q1, med, q3 = np.percentile(df_features[df_features['record_ID']==rec].iloc[:,col], [25, 50, 75])
            iqr = q3 - q1
            loval = q1 - 1.5 * iqr
            hival = q3 + 1.5 * iqr
            for row in range(df_features[df_features['record_ID']==rec].shape[0]):
                #if np.abs(normed_features[row]) > 3.5:
                if df_features[df_features['record_ID']==rec].iloc[row, col] < loval or df_features[df_features['record_ID']==rec].iloc[row,col] > hival:
                    ids_to_remove.append(df_features[df_features['record_ID']==rec].iloc[row].name)
    '''
    a = 0

