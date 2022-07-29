import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('once')

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
from utils import robust_z_score_norm, minmax_norm
import multiprocessing

import pandas as pd
from pdpbox import pdp, get_dataset, info_plots
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    ## PARAMETERS ##
    data_path = '/home/jack/cluster_results/whole_features_data_30sec_fixed.csv'
    path_n_windows = '/home/jack/cluster_results/n_windows/whole_n_windows_data_30sec.txt'
    split_a_test_set = True
    subj_to_use = 'SA047' #'all'
    discrim_feature = 'deltaPANSS_posit'
    normalize = False
    directory_targets = '/home/jack/cluster_results/labels'
    ################
    # Load data
    datas = pd.read_csv(data_path, index_col=0)
    datas = datas.drop(columns=['PANSS', 'PANSS_posit'])  # remove target columns, the specified
                                                   # feature target will be added later
    # Add record_id info
    df_n_windows = pd.read_csv(path_n_windows, index_col=0, header=None)
    df_n_windows = df_n_windows.to_dict()
    df_n_windows = df_n_windows[1]
    new_col = []
    for key, value in df_n_windows.items():
        new_segment = value * [key]
        new_col = [*new_col, *new_segment]

    datas['record_ID'] = new_col
    for i in range(datas.shape[0]): # sanity check
        assert datas['record_ID'][i][:5] == datas['SUBJ'][i]

    # Conserve only the subject that we wants
    datas = datas[datas['SUBJ'] == subj_to_use]
    datas = datas.drop(columns=['SUBJ'])  # Now the SUBJ info is reduntant with record_ID info,
                            # and probably we are using one subject at once

    for i in range(datas.shape[0]): # sanity check
        assert datas['record_ID'].iloc[i].startswith(subj_to_use)

    # Target info
    target_info = pd.read_csv('{}/{}_labels.csv'.format(directory_targets, subj_to_use), index_col=0, decimal=',')
    target_info = target_info.to_dict()
    target_info = target_info[discrim_feature]
    target_col = []
    for r in range(datas.shape[0]):
        target_col.append(target_info[datas['record_ID'].iloc[r]])
    datas['target'] = target_col
    datas = datas.dropna()
    datas = datas.drop(columns=['record_ID'])    # We already have the labels value for each window, the record_IDs are not necessarily anymore

    # Split data in train and test (if requested)
    if split_a_test_set:
        X_, X_test, y_, y_test = train_test_split(datas.drop(columns='target'),
                                                                datas['target'])
        X_test = X_test.values
        y_test = y_test.values
    else:
        X_ = datas.drop(columns='target')
        y_ = datas['target']
    X_columns_name = X_.columns
    X_ = X_.values
    y_ = y_.values

    # Normalization
    if normalize:
        temp_X_norm = np.zeros(X_.shape)
        for f_id in range(X_.shape[1]):
            temp_X_norm[:,f_id] = robust_z_score_norm(X_[:,f_id])
        X_ = temp_X_norm
        y_ = minmax_norm(y_)

    # Create model
    model = RandomForestRegressor(
            n_estimators = 100,
            criterion    = 'mse',# 'squared_error',#'mse',
            max_depth    = 2,
            max_features =5,
            random_state = 123
         )
    # Train model
    model = model.fit(X_, y_)
    #a = 0
    '''
    predictions = model.predict(X=X_test)
    rmse = mean_squared_error(
        y_true=y_test,
        y_pred=predictions,
        squared=False
    )
    print(f"El error (rmse) de test es: {rmse}")
    '''
    print(model.score(X_,y_))
    print(model.score(X_test, y_test))
    # Importancia nodal
    predictors_importance = pd.DataFrame(
        {'predictor': datas.drop(columns='target').columns,
         'importance': model.feature_importances_}
    )
    print("Importancia de los predictores en el modelo")
    print("-------------------------------------------")
    predictors_importance.sort_values('importance', ascending=False)
    print(predictors_importance.sort_values('importance', ascending=False).head(10))
    print()
    # Importancia por permutación
    importance = permutation_importance(
        estimator=model,
        X=X_,
        y=y_,
        n_repeats=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=multiprocessing.cpu_count() - 1,
        random_state=123
    )
    print("\nImportancia por permutación")
    print("-------------------------------------------")

    # Se almacenan los resultados (media y desviación) en un dataframe
    df_importance = pd.DataFrame(
        {k: importance[k] for k in ['importances_mean', 'importances_std']}
    )
    df_importance['feature'] = X_columns_name
    df_importance.sort_values('importances_mean', ascending=False)
    print(df_importance.sort_values('importances_mean', ascending=False).head(10))
    print('Target feature: ' + str(discrim_feature))

    a=0
