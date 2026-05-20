from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import List
import pandas as pd

def naive_bayes(df: pd.DataFrame, features: List[str], target: List[str]):
    X=df[features]
    y=df[target]
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    model= GaussianNB()
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    return classification_report(y_test,y_pred, output_dict=True)