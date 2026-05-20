from sklearn.linear_model import LinearRegression
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

def linear_regression(df: pd.DataFrame, features: List[str], target: List[str]):
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # pca= PCA(n_components=1)
    # X_train_pca= pca.fit_transform(X_train_scaled)
    # X_test_pca=pca.fit(X_test_scaled)

    model= LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred=model.predict(X_test_scaled)

    # residuals = y_test - y_pred
    
    # plt.figure(figsize=(8,5))
    # plt.scatter(X_test_pca, y_test, label='Actual Values')
    # plt.plot(X_test_pca, y_pred, linewidth=2, label='Regression Line')

    # plt.xlabel('Principal Component 1')
    # plt.ylabel(target)
    # plt.title('PCA + Linear Regression')
    # plt.legend()
    # plt.show()



    # plt.figure(figsize=(8,5))

    # plt.scatter(y_pred, residuals)
    # plt.axhline(y=0, linestyle='--')

    # plt.xlabel('Predicted Values')
    # plt.ylabel('Residuals')
    # plt.title('Residual Plot')

    # plt.show()


    metrics_df=pd.DataFrame({
        'Metrics':[
            'R2 Score',
            'MAE',
            'MSE',
            'RMSE'
        ],
        'Values':[
            r2_score(y_test,y_pred),
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred),
            np.sqrt(mean_squared_error(y_test, y_pred))
        ]
    })
    return metrics_df



    