from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from .data_cleaning import clean_dataset

def support_vector_machine(df: pd.DataFrame, selected_features: list[str], selected_target: list[str]):
    df = clean_dataset(df)
    X = df[selected_features]
    y = df[selected_target]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create an SVM model
    model = svm.SVC(kernel='linear', random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.decision_function(X_test_scaled)

    return classification_report(y_test,y_pred, output_dict=True), y_pred_prob, y_test, y, y_pred

