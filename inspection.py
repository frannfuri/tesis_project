import numpy as np
import pandas as pd
from utils import robust_z_score_norm

df = pd.read_csv('/home/jack/cluster_results/whole_features_data_60sec_fixed.csv', index_col=0)
path_n_windows = '/home/jack/cluster_results/n_windows/whole_n_windows_data_60sec.txt'
directory_targets = '/home/jack/cluster_results/labels'

df = df.drop(columns=['PANSS', 'PANSS_posit', 'IAF_post_chns','AlphaBand_high_bound','AlphaBand_low_bound', 'CoG_post_chns'])  # remove target columns, the specified
                                                                # feature target will be added later
# Add record_id info
df_n_windows = pd.read_csv(path_n_windows, index_col=0, header=None)
df_n_windows = df_n_windows.to_dict()
df_n_windows = df_n_windows[1]
new_col = []
for key, value in df_n_windows.items():
    new_segment = value * [key]
    new_col = [*new_col, *new_segment]

df['record_ID'] = new_col
list_of_records = df['record_ID'].unique()

# Normalization of features to compare stds later
all_subjs_names = df['SUBJ'].unique()
all_df_subjs = []
for s in all_subjs_names:
    df_subj = df[df['SUBJ']==s]
    column_names = list(df_subj.keys())
    for feat in column_names:
        if feat != 'SUBJ' and feat != 'record_ID':
            df_subj[feat] = robust_z_score_norm(df_subj[feat])
    all_df_subjs.append(df_subj)

standarized_df = pd.concat(all_df_subjs, axis=0)
df = standarized_df

assert standarized_df.isna().sum().sum() == 0

s007_feats = []
s010_feats = []
s014_feats = []
s017_feats = []
s047_feats = []
for rec in list_of_records:
    all_feat_desvs = []
    for c in range(1, df.shape[1] - 1):
        desv = np.std(df[df['record_ID'] == rec].iloc[:, c])
        all_feat_desvs.append(desv)
    max_ids = np.argsort(all_feat_desvs)[:20]
    print('RECORD: {}'.format(rec))
    print('features with less desviation: {}'.format(df.keys()[max_ids]))
    if rec.startswith('SA007'):
        s007_feats.append(set(df.keys()[max_ids]))
    elif rec.startswith('SA010'):
        s010_feats.append(set(df.keys()[max_ids]))
    elif rec.startswith('SA014'):
        s014_feats.append(set(df.keys()[max_ids]))
    elif rec.startswith('SA017'):
        s017_feats.append(set(df.keys()[max_ids]))
    elif rec.startswith('SA047'):
        s047_feats.append(set(df.keys()[max_ids]))
    else:
        assert 0 == 1
print('==========================')
print('SA007 intersection -->  {}'.format(set.intersection(*s007_feats)))
print('SA010 intersection -->  {}'.format(set.intersection(*s010_feats)))
print('SA014 intersection -->  {}'.format(set.intersection(*s014_feats)))
print('SA017 intersection -->  {}'.format(set.intersection(*s017_feats)))
print('SA047 intersection -->  {}'.format(set.intersection(*s047_feats)))

a = 0
