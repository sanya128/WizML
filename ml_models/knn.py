from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import List
import pandas as pd

def k_neighbor(df: pd.DataFrame, features: List[str], target: List[str]):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    y_pred=model.predict(X_test_scaled)
    return classification_report(y_test,y_pred, output_dict=True)


