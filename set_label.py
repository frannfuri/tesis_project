import pandas as pd
import os

if __name__ == '__main__':
    path_of_file = '/home/jack/cluster_results/whole_features_data_30sec_fixed.csv'
    path_n_windows = '/home/jack/cluster_results/n_windows/whole_n_windows_data_30sec.txt'
    target_to_consider = 'deltaSAPS'
    directory_targets = '/home/jack/cluster_results/labels'

    df = pd.read_csv(path_of_file, index_col=0)
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
    for i in range(df.shape[0]): # sanity check
        assert df['record_ID'][i][:5] == df['SUBJ'][i]

    all_target_info = {}
    for root, _, txt_files in os.walk(directory_targets):
        for txt_f in sorted(txt_files):
            target_info = pd.read_csv('{}/{}'.format(root, txt_f), index_col=0, decimal=',')
            target_info = target_info.to_dict()
            all_target_info.update(target_info[target_to_consider])
    target_col = []
    for r in range(df.shape[0]):
        target_col.append(all_target_info[df['record_ID'].iloc[r]])
    df['target'] = target_col

    df = df.drop(columns=['SUBJ', 'record_ID'])
    df.to_csv('./analysis_features_{}_target_{}.csv'.format(path_of_file[-15:-10], target_to_consider))