import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.cm as cmx

if __name__ == '__main__':
    feat1 = 'std_Alpha'
    feat2 = 'std_Delta'
    feat3 = 'std_Theta'
    subj_name = 'SA007'
    discrim_f = 'deltaPANSS_posit'
    # Window length dependents
    path_csv = '/home/jack/cluster_results/whole_features_data_60sec_fixed.csv'
    path_n_windows = '/home/jack/cluster_results/n_windows/whole_n_windows_data_60sec.txt'

    directory_targets = '/home/jack/cluster_results/labels'
    colorsMap = 'inferno'
    ###################################

    # Preprocess dataframe
    df = pd.read_csv(path_csv, index_col=0)
    df = df.drop(columns=['PANSS', 'PANSS_posit'])  # remove target columns, the specified
                                                # feature target will be added later
    df_n_windows = pd.read_csv(path_n_windows, index_col=0, header=None)
    df_n_windows = df_n_windows.to_dict()
    df_n_windows = df_n_windows[1]
    new_col = []
    for key, value in df_n_windows.items():
        new_segment = value * [key]
        new_col = [*new_col, *new_segment]

    df['record_ID'] = new_col
    for i in range(df.shape[0]):  # sanity check
        assert df['record_ID'][i][:5] == df['SUBJ'][i]

    all_target_info = {}
    for root, _, txt_files in os.walk(directory_targets):
        for txt_f in sorted(txt_files):
            target_info = pd.read_csv('{}/{}'.format(root, txt_f), index_col=0, decimal=',')
            target_info = target_info.to_dict()
            all_target_info.update(target_info[discrim_f])
    target_col = []
    for r in range(df.shape[0]):
        target_col.append(all_target_info[df['record_ID'].iloc[r]])
    df['target'] = target_col

    df = df[df['SUBJ']==subj_name]
    df = df.drop(columns=['SUBJ'])
    list_of_records = df['record_ID'].unique()
    for rec in list_of_records:             # Sanity checks!
        assert np.median(df['target'][df['record_ID'] == rec]) ==\
                                        df['target'][df['record_ID'] == rec].iloc[0]
        assert all(t_val == df[df['record_ID'] == rec].iloc[0]['target'] for t_val in
                                        df[df['record_ID'] == rec]['target'])

    # Plot
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(df['target'])-2, vmax=max(df['target'])+2)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df[feat1], df[feat2], df[feat3], c=scalarMap.to_rgba(df['target']))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-1, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
    ax.ticklabel_format(axis='z', style='sci', scilimits=(-1, 2))
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_zlabel(feat3)
    scalarMap.set_array(df['target'])
    fig.colorbar(scalarMap, label=discrim_f)
    plt.show(block=False)

    a = 0