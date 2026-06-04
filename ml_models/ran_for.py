from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .data_cleaning import clean_dataset


def random_forest_classifier(df: pd.DataFrame, features: List[str], target:List[str]):
    df = clean_dataset(df)
    X= df[features]
    y= df[target]
    #le=LabelEncoder()
    #y["targets"]=le.fit_transform(y["targets"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model= RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    y_pred_prob = model.predict_proba(X_test)

    return classification_report(y_test,y_pred, output_dict=True), y_pred_prob, y_test, y, y_pred
