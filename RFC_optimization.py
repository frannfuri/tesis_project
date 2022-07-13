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
import multiprocessing

def estimate_numb_trees_oob(X_train, y_train, range_of_estimators=150):
    # Validation with Out-of-Bag error
    train_scores = []
    oob_scores = []
    # Evaluated values
    estimator_range = range(1, range_of_estimators, 5)
    # Train each model and extract its train error and from Out-of-Bag
    for n_estimators in estimator_range:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion='mse',
            max_depth=None,
            max_features='auto',
            oob_score=True,
            n_jobs=-1,
            random_state=123
        )
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        oob_scores.append(model.oob_score_)
    # Graph with the errors evolution
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(estimator_range, train_scores, label="train scores")
    ax.plot(estimator_range, oob_scores, label="out-of-bag scores")
    ax.plot(estimator_range[np.argmax(oob_scores)], max(oob_scores),
            marker='o', color="red", label="max score")
    ax.set_ylabel("R^2")
    ax.set_xlabel("n_estimators")
    ax.set_title("Evolución del out-of-bag-error vs número árboles")
    plt.legend();
    print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(oob_scores)]}")

def estimate_numb_trees_cv(X_train, y_train, range_of_estimators=150):
    # Validación empleando k-cross-validation y neg_root_mean_squared_error
    # ==============================================================================
    train_scores = []
    cv_scores = []

    # Valores evaluados
    estimator_range = range(1, range_of_estimators, 5)

    # Bucle para entrenar un modelo con cada valor de n_estimators y extraer su error
    # de entrenamiento y de k-cross-validation.
    for n_estimators in estimator_range:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion='mse',
            max_depth=None,
            max_features='auto',
            oob_score=False,
            n_jobs=-1,
            random_state=123
        )

        # Error de train
        model.fit(X_train, y_train)
        predictions = model.predict(X=X_train)
        rmse = mean_squared_error(
            y_true=y_train,
            y_pred=predictions,
            squared=False
        )
        train_scores.append(rmse)

        # Error de validación cruzada
        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            scoring='neg_root_mean_squared_error',
            cv=5
        )
        # Se agregan los scores de cross_val_score() y se pasa a positivo
        cv_scores.append(-1 * scores.mean())

    # Gráfico con la evolución de los errores
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(estimator_range, train_scores, label="train scores")
    ax.plot(estimator_range, cv_scores, label="cv scores")
    ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores),
            marker='o', color="red", label="min score")
    ax.set_ylabel("root_mean_squared_error")
    ax.set_xlabel("n_estimators")
    ax.set_title("Evolución del cv-error vs número árboles")
    plt.legend()
    print(f"Valor óptimo de n_estimators: {estimator_range[np.argmin(cv_scores)]}")

def estimated_max_features_oob(X_train, y_train):
    # Validación empleando el Out-of-Bag error
    # ==============================================================================
    train_scores = []
    oob_scores = []

    # Valores evaluados
    max_features_range = range(1, X_train.shape[1] + 1, 1)

    # Bucle para entrenar un modelo con cada valor de max_features y extraer su error
    # de entrenamiento y de Out-of-Bag.
    for max_features in max_features_range:
        model = RandomForestRegressor(
            n_estimators=100,
            criterion= 'mse',
            max_depth=None,
            max_features=max_features,
            oob_score=True,
            n_jobs=-1,
            random_state=123
        )
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        oob_scores.append(model.oob_score_)

    # Gráfico con la evolución de los errores
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(max_features_range, train_scores, label="train scores")
    ax.plot(max_features_range, oob_scores, label="out-of-bag scores")
    ax.plot(max_features_range[np.argmax(oob_scores)], max(oob_scores),
            marker='o', color="red")
    ax.set_ylabel("R^2")
    ax.set_xlabel("max_features")
    ax.set_title("Evolución del out-of-bag-error vs número de predictores")
    plt.legend()
    print(f"Valor óptimo de max_features: {max_features_range[np.argmax(oob_scores)]}")

def estimate_max_features_cv(X_train, y_train):
    # Validación empleando k-cross-validation y neg_root_mean_squared_error
    # ==============================================================================
    train_scores = []
    cv_scores = []

    # Valores evaluados
    max_features_range = range(1, X_train.shape[1] + 1, 1)

    # Bucle para entrenar un modelo con cada valor de max_features y extraer su error
    # de entrenamiento y de k-cross-validation.
    for max_features in max_features_range:
        model = RandomForestRegressor(
            n_estimators=100,
            criterion='mse',
            max_depth=None,
            max_features=max_features,
            oob_score=True,
            n_jobs=-1,
            random_state=123
        )

        # Error de train
        model.fit(X_train, y_train)
        predictions = model.predict(X=X_train)
        rmse = mean_squared_error(
            y_true=y_train,
            y_pred=predictions,
            squared=False
        )
        train_scores.append(rmse)

        # Error de validación cruzada
        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            scoring='neg_root_mean_squared_error',
            cv=5
        )
        # Se agregan los scores de cross_val_score() y se pasa a positivo
        cv_scores.append(-1 * scores.mean())

    # Gráfico con la evolución de los errores
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(max_features_range, train_scores, label="train scores")
    ax.plot(max_features_range, cv_scores, label="cv scores")
    ax.plot(max_features_range[np.argmin(cv_scores)], min(cv_scores),
            marker='o', color="red", label="min score")
    ax.set_ylabel("root_mean_squared_error")
    ax.set_xlabel("max_features")
    ax.set_title("Evolución del cv-error vs número de predictores")
    plt.legend();
    print(f"Valor óptimo de max_features: {max_features_range[np.argmin(cv_scores)]}")

def grid_search_oob(X_train, y_train, n_estimators, max_features, max_depth):
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = ParameterGrid(
        {'n_estimators': n_estimators, # [150]
         'max_features': max_features, # [5, 7, 9]
         'max_depth': max_depth # [None, 3, 10 , 20]
         }
    )

    # Loop para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    resultados = {'params': [], 'oob_r2': []}

    for params in param_grid:
        model = RandomForestRegressor(
            oob_score=True,
            n_jobs=-1,
            random_state=123,
            **params
        )

        model.fit(X_train, y_train)

        resultados['params'].append(params)
        resultados['oob_r2'].append(model.oob_score_)
        print(f"Modelo: {params} \u2713")

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('oob_r2', ascending=False)
    resultados.head(4)
    return resultados

def grid_search_cv(X_train, y_train, n_estimators, max_features, max_depth):
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = {'n_estimators': n_estimators, # [150]
                  'max_features': max_features, #[5, 7, 9],
                  'max_depth': max_depth #  [None, 3, 10, 20]
                  }

    # Búsqueda por grid search con validación cruzada
    # ==============================================================================
    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=123),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        n_jobs=multiprocessing.cpu_count() - 1,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
        refit=True,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X=X_train, y=y_train)

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(grid.cv_results_)
    resultados.filter(regex='(param.*|mean_t|std_t)') \
        .drop(columns='params') \
        .sort_values('mean_test_score', ascending=False) \
        .head(4)
    return resultados, grid

