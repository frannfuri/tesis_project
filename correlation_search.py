import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from pygam import LinearGAM
import numpy as np

path_features = '/home/jack/cluster_results/whole_features_data_30sec_fixed.csv'
path_n_windows = '/home/jack/cluster_results/n_windows/whole_n_windows_data_30sec.txt'
target_feat = 'deltaPANSS_posit'
consider_sbj = 'all' #'SA047' # if all subjects will be considered, then set "all"

directory_targets = '/home/jack/cluster_results/labels'
split_test_set = False
try_outliers_detection = True
###########################

# Load data
df_features = pd.read_csv(path_features, index_col=0)
df_features = df_features.drop(columns=['PANSS', 'PANSS_posit']) # remove target columns, the specified
                                                                    # feature target will be added later

# Add record_id info
df_n_windows = pd.read_csv(path_n_windows, index_col=0,header=None)
df_n_windows = df_n_windows.to_dict()
df_n_windows= df_n_windows[1]
new_col = []
for key, value in df_n_windows.items():
    new_segment = value * [key]
    new_col = [*new_col, *new_segment]

df_features['record_ID'] = new_col

for i in range(df_features.shape[0]): # sanity check
    assert df_features['record_ID'][i][:5] == df_features['SUBJ'][i]

df_features = df_features.drop(columns=['SUBJ']) # now the SUBJ info is redundant with record_ID info

if consider_sbj != 'all': # revisar solo los labels del sujeto en consideración
    df_features = df_features[df_features['record_ID'].str.contains(consider_sbj)]
    target_info = pd.read_csv('{}/{}_labels.csv'.format(directory_targets, consider_sbj), index_col=0, decimal=',')
    target_info = target_info.to_dict()
    target_info = target_info[target_feat]
    target_col = []
    for r in range(df_features.shape[0]):
        target_col.append(target_info[df_features['record_ID'].iloc[r]])
    df_features['target'] = target_col

else: # revisar todos los files de labels
    all_target_info = {}
    for root, _, txt_files in os.walk(directory_targets):
        for txt_f in sorted(txt_files):
            target_info = pd.read_csv('{}/{}'.format(root, txt_f), index_col=0, decimal=',')
            target_info = target_info.to_dict()
            all_target_info.update(target_info[target_feat])
    target_col = []
    for r in range(df_features.shape[0]):
        target_col.append(all_target_info[df_features['record_ID'].iloc[r]])
    df_features['target'] = target_col

df_features = df_features.dropna()
y = df_features['target'].values
df_features = df_features.drop(columns=['record_ID', 'target']) # From now, this are unnecessary columns
X_columns_names = df_features.columns
X = df_features.values
if try_outliers_detection:
    from sklearn.ensemble import IsolationForest

    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(X, y)
    outliers = isf.predict(X)
    X = np.delete(X, [i for i in range(len(outliers)) if outliers[i] == -1], axis=0)
    y = np.delete(y, [i for i in range(len(outliers)) if outliers[i] == -1])

if split_test_set:
    X, X_test, y, y_test = train_test_split(X, y, random_state=123)

from xgboost import XGBRegressor

m = XGBRegressor(
    #objective='reg:linear',
    max_depth=2,
    gamma=2,
    learning_rate=0.8, #eta
    reg_alpha=0.5,
    reg_lambda=0.5
)
m.fit(X, y)
print('Train final score: {}'.format(m.score(X,y)))
if split_test_set:
    print('Test final score: {}'.format(m.score(X_test,y_test)))
a = 0