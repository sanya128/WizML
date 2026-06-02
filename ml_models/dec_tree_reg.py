from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from typing import List
import pandas as pd
import numpy as np

def dec_tree_reg(df: pd.DataFrame, features: List[str], target: List[str]):

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(features) > 1:
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        X_train_pca = X_train.values if hasattr(X_train, 'values') else X_train
        X_train_pca = X_train_pca.reshape(-1, 1) if X_train_pca.ndim == 1 else X_train_pca
        X_test_pca = X_test.values if hasattr(X_test, 'values') else X_test
        X_test_pca = X_test_pca.reshape(-1, 1) if X_test_pca.ndim == 1 else X_test_pca

    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)

    y_test_flat = y_test.values.ravel() if hasattr(y_test, 'values') else y_test.ravel()
    residuals = y_test_flat - y_pred

    z = np.polyfit(X_test_pca[:, 0], y_test_flat, 1)
    p = np.poly1d(z)
    bestfit_line = p(np.sort(X_test_pca[:, 0]))

    metrics_df = pd.DataFrame({
        'Metrics':[
            'R2 Score',
            'MAE',
            'MSE',
            'RMSE'
        ],
        'Values':[
            r2_score(y_test, y_pred),
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred),
            np.sqrt(mean_squared_error(y_test, y_pred))
        ]
    })

    pca_data = {
        'X_test_pca': X_test_pca,
        'y_test': y_test_flat,
        'y_pred': y_pred,
        'residuals': residuals,
        'bestfit_line': bestfit_line,
        'bestfit_x': np.sort(X_test_pca[:, 0])
    }

    return metrics_df, pca_data



    