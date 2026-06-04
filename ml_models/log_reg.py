from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from typing import List, Any
import pandas as pd
import numpy as np
from .data_cleaning import clean_dataset 

# class LogRegression:
#     def __init__(self, df: pd.DataFrame,  features: List[str], target: List[str]):
#         self.df = df 
#         self.features = features 
#         self.target = target 
#         self.X = self.df[features]
#         self.y = self.df[target] 
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.X, self.y, test_size=0.2, random_state=42, stratify=y
#         )
#     def fit(self):
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(self.X_train)
#         X_test_scaled = scaler.transform(self.X_test)

#         model= LogisticRegression()
#         model.fit(X_train_scaled, self.y_train)
#         self.y_pred = model.predict(X_test_scaled)
#         return classification_report(self.y_test,y_pred, output_dict=True)

def logistic_regression(df: pd.DataFrame, features: List[str], target: List[str]) -> tuple[str | dict, np.ndarray, np.ndarray]:
    '''
    Running Logistic Regression on df
    @param df: pd.DataFrame object
    '''
    df = clean_dataset(df)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model= LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred= model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)

    return classification_report(y_test,y_pred, output_dict=True), y_pred_prob, y_test, y, y_pred