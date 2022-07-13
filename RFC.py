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
    data_path = 'results_bef_13-07/whole_selected_features_data_30sec_standard.csv'
    split_a_test_set = True
    discrim_feature = 'PANSS'
    features_to_drop = ['SUBJ', 'CoG_post_chns', 'AlphaBand_low_bound', 'AlphaBand_high_bound','PANSS_posit']
    normalize = False
    ################
    # Load data
    datas = pd.read_csv(data_path, index_col=0)
    datas = datas.drop(columns=features_to_drop)
    # Split data in train and test (if requested)
    if split_a_test_set:
        X_, X_test, y_, y_test = train_test_split(datas.drop(columns=discrim_feature),
                                                                datas[discrim_feature], random_state=123)
        X_test = X_test.values
        y_test = y_test.values
    else:
        X_ = datas.drop(columns=discrim_feature)
        y_ = datas[discrim_feature]
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
            n_estimators = 200,#100,#100, #100
            criterion    = 'mse',# 'squared_error',#'mse',
            max_depth    = None,
            max_features =25,#26,# 30, #'auto',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )
    # Train model
    model.fit(X_, y_)
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
        {'predictor': datas.drop(columns=discrim_feature).columns,
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

    ross_data = datas
    ross_features = list(X_columns_name)
    ross_model = model
    ross_target = discrim_feature
    a=0
