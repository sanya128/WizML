import streamlit as st 
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from WizML.ml_models.log_reg import logistic_regression
from WizML.ml_models.ran_for import random_forest_classifier
import math 

try:

    st.title("Machine Learning Dashboard")

    #==================================================
    # memory stuffs
    if "data" not in st.session_state:
        st.session_state.data = None

    #==================================================

    select_method=st.sidebar.selectbox(
        label = 'Select the type of algorithms',
        options = ('Regression', 'Classification'),
        index=None,
        placeholder="Select"
    )
    if(select_method=='Classification'):
        dataset_select=st.sidebar.selectbox(
            label = 'Select the dataset',
            options = ('Iris', 'Heart Disease'),
            index=None,
            placeholder="Select"
        )
        model_select=st.sidebar.selectbox(
            label = 'Select the algorithm',
            options = ('Logistic Regression','Random Forest','Decision Tree', 'Support Vector Machine', 'Naive Bayes' ,'K-Nearest Neighbors'),
            index=None,
            placeholder="Select"
        )
    else:
        dataset_select=st.sidebar.selectbox(
            label = 'Select the dataset',
            options = ('Air Quality', 'Wine Quality'),
            index=None,
            placeholder="Select"
        )

        model_select=st.sidebar.selectbox(
            label = 'Select the algorithm',
            options = ('Linear Regression','Ridge & Lasso Regression','Support Vector Regression', 'Random Forest Regression'),
            index=None,
            placeholder="Select"
        )

    if(dataset_select=='Iris'):
        iris = fetch_ucirepo(id=53) 
        X = iris.data.features 
        y = iris.data.targets 
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
        st.dataframe(st.session_state.data, hide_index='True')

    if(dataset_select=='Heart Disease'):
        heart_disease = fetch_ucirepo(id=45) 
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
        st.dataframe(st.session_state.data, hide_index='True')

    if(dataset_select=='Air Quality'):
        air_quality = fetch_ucirepo(id=360)  
        X = air_quality.data.features 
        y = air_quality.data.targets
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
        st.dataframe(st.session_state.data, hide_index='True')

    if(dataset_select=='Wine Quality'):
        wine_quality = fetch_ucirepo(id=186) 
        X = wine_quality.data.features 
        y = wine_quality.data.targets 
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
        st.dataframe(st.session_state.data, hide_index='True')

    selected_features = st.multiselect(
        "Select your features",
        options=list(st.session_state.data.columns),
    )
    selected_target= st.multiselect(
        "Select your target",
        options=list(st.session_state.data.columns)
    )
    if(model_select=='Logistic Regression'):
        classification_report=logistic_regression(st.session_state.data, selected_features, selected_target)
        classification_dataframe = pd.DataFrame(classification_report)
        accuracy = classification_dataframe["accuracy"].unique()[0]
        classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
        st.dataframe(classification_dataframe)
        st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

    if(model_select=='Random Forest'):
        classification_report= random_forest_classifier(st.session_state.data, selected_features, selected_target)
        classification_dataframe = pd.DataFrame(classification_report)
        accuracy = classification_dataframe["accuracy"].unique()[0]
        classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
        st.dataframe(classification_dataframe)
        st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

    

except:
    pass
