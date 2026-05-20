import streamlit as st 
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from ml_models.log_reg import logistic_regression
from ml_models.lin_reg import linear_regression
from ml_models.ran_for import random_forest_classifier
from ml_models.svm import support_vector_machine
from ml_models.n_bayes import naive_bayes
from ml_models.knn import k_neighbor
from ml_models.dec_tree import dec_tree
from ml_models.dec_tree_reg import dec_tree_reg
import streamlit.components.v1 as components
import os
import math 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def welcome_page():
    st.set_page_config(page_title="WizML", layout="centered")

    
    st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        .stApp { background: #080810; }
        /* old theme */
        /* .stApp { background: #000000; } */
        .block-container { padding: 0 !important; max-width: 100% !important; }
        /* Hide visually but keep clickable in DOM */
        div[data-testid="stButton"] {
            position: absolute;
            width: 1px;
            height: 1px;
            overflow: hidden;
            opacity: 0;
            pointer-events: none;
        }
    </style>
    """, unsafe_allow_html=True)
    # Force black background, hide Streamlit chrome
    # st.markdown("""
    # <style>
    #     #MainMenu, footer, header { visibility: hidden; }
    #     .stApp { background: #000000; }
    #     .block-container { padding: 0 !important; max-width: 100% !important; }
    # </style>
    # """, unsafe_allow_html=True)



    # Listen for postMessage from the HTML iframe
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'enter_platform') {
            window.top.location.href = window.top.location.pathname + '?enter=true';
        }
    });
    </script>
    """, unsafe_allow_html=True)

    if st.button("Enter Platform", key="enter_btn"):
        st.session_state.logged_in = True
        st.rerun()

    
    BASE_DIR = os.path.dirname(__file__)
    html_path = os.path.join(BASE_DIR, "html_pages/wlcm.html")
    with open(html_path, "r") as f:
        html_content = f.read()

    components.html(html_content, height=600, scrolling=False)
    

def show_main_platform():

    st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        .stApp { background: #080810; color: #b4d4f0; }
        .stSidebar { background: #0a0e18; }
        section[data-testid="stSidebar"] * { color: #b4d4f0 !important; }
        .stSelectbox label { color: #47aaff !important; }
        .stSelectbox > div > div { background: #0d1520 !important; color: #b4d4f0 !important; border-color: #1a3550 !important; }
        .stMultiSelect label { color: #47aaff !important; }
        .stMultiSelect > div > div { background: #0d1520 !important; color: #b4d4f0 !important; border-color: #1a3550 !important; }
        
        .stExpander { border-color: #2a2a4a !important; background: #1a1a2e !important; }
        .stExpander summary { color: #7eb8f7 !important; }
        .stDataFrame { background: #1a1a2e !important; }
        /* force expander body and all children */
        details { background: #1a1a2e !important; }
        details > div { background: #1a1a2e !important; }
        [data-testid="stExpanderDetails"] { background: #1a1a2e !important; }
        [data-testid="stExpanderDetails"] > div { background: #1a1a2e !important; }
        /* force the white expander header bar */
        details summary { background: #1a1a2e !important; border-color: #2a2a4a !important; }
                
        /* .stExpander { border-color: #1a3550 !important; background: #0a0e18 !important; } */
        /* .stExpander summary { color: #47aaff !important; } */
        /* .stDataFrame { background: #0a0e18 !important; } */
        .stMetric label, .stMetric div { color: #b4d4f0 !important; }
        h1, h2, h3, p, label { color: #b4d4f0 !important; }
                
        .block-container { padding-top: 2rem; }
        /* Expander header text */
        .stExpander summary p { color: #47aaff !important; }
        .stExpander summary svg { fill: #47aaff !important; }
        /* Dataframe table */
        .stDataFrame thead th { background: #0d1520 !important; color: #47aaff !important; }
        .stDataFrame tbody td { background: #080810 !important; color: #b4d4f0 !important; }
        .stDataFrame tbody tr:hover td { background: #0d1520 !important; }
        /* Dataframe scrollbar and borders */
        .stDataFrame * { border-color: #1a3550 !important; }
        /* dataframe index column */
        .stDataFrame th:first-child { color: #1a8cff !important; }
        /* book loader iframe background */
        iframe { background: #080810 !important; }
                
        /* multiselect tags */
        .stMultiSelect span[data-baseweb="tag"] { background: #0d1520 !important; color: #47aaff !important; border: 1px solid #1a3550 !important; }
        /* dropdown options */
        li[role="option"] { background: #0d1520 !important; color: #b4d4f0 !important; }
        li[role="option"]:hover { background: #1a3550 !important; }
    </style>
    """, unsafe_allow_html=True)

    #old theme
    # st.markdown("""
    # <style>
    #     #MainMenu, footer, header { visibility: hidden; }
    #     .stApp { background: #000000; color: #ffffff; }
    #     .stSidebar { background: #0a0a0a; }
    #     section[data-testid="stSidebar"] * { color: #ffffff !important; }
    #     .stSelectbox label { color: #ffffff !important; }
    #     .stSelectbox > div > div { background: #111 !important; color: #fff !important; border-color: #333 !important; }
    #     .stMultiSelect label { color: #ffffff !important; }
    #     .stMultiSelect > div > div { background: #111 !important; color: #fff !important; border-color: #333 !important; }
    #     .stExpander { border-color: #333 !important; background: #0d0d0d !important; }
    #     .stExpander summary { color: #ffffff !important; }
    #     .stDataFrame { background: #0d0d0d !important; }
    #     .stMetric label, .stMetric div { color: #ffffff !important; }
    #     h1, h2, h3, p, label { color: #ffffff !important; }
    #     .block-container { padding-top: 2rem; }
    # </style>
    # """, unsafe_allow_html=True)

    # Pixel title
    BASE_DIR = os.path.dirname(__file__)
    title_html = os.path.join(BASE_DIR, "html_pages/pixel_title.html")
    with open(title_html, "r") as f:
        title_content = f.read()
    components.html(title_content, height=60, scrolling=False)
    # title_html = os.path.join(BASE_DIR, "html_pages/pixel_title.html")
    # with open(title_html, "r") as f:
    #     components.html(f.read(), height=100, scrolling=False)

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
            options = ('Linear Regression','Ridge & Lasso Regression', 'Decision Tree Regression', 'Support Vector Regression', 'Random Forest Regression'),
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


    if(dataset_select=='Heart Disease'):
        heart_disease = fetch_ucirepo(id=45) 
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
        

    if(dataset_select=='Air Quality'):
        air_quality = fetch_ucirepo(id=360)  
        X = air_quality.data.features 
        y = air_quality.data.targets
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
        

    if(dataset_select=='Wine Quality'):
        wine_quality = fetch_ucirepo(id=186) 
        X = wine_quality.data.features 
        y = wine_quality.data.targets 
        df=pd.DataFrame(data=X)
        df['target']=y
        st.session_state.data = df
    with st.expander(label="View Dataset", expanded=True):
        if st.session_state.data is None:
            st.markdown("<span style='color:#555'>No dataset loaded yet.</span>", unsafe_allow_html=True)
        else:
            # Show book animation while loading
            book_html_path = os.path.join(BASE_DIR, "html_pages/book_loader.html")
            with open(book_html_path, "r") as f:
                book_content = f.read()
            components.html(book_content, height=220, scrolling=False)
            # with open(book_html_path, "r") as f:
            #     components.html(f.read(), height=220, scrolling=False)

            # inject dataframe theme via markdown
            styled_df = st.session_state.data.style.set_properties(**{
                'background-color': '#1a1a2e',
                'color': '#c8d8f0',
                'border-color': '#2a2a4a',
                'font-family': 'Courier New, monospace',
                'font-size': '13px'
            }).set_table_styles([
                {'selector': 'thead th', 'props': [
                    ('background-color', '#0f3460'),
                    ('color', '#a8d8ff'),
                    ('border-color', '#2a2a4a'),
                    ('font-family', 'Courier New, monospace'),
                    ('font-size', '13px'),
                    ('padding', '8px'),
                    ('font-weight', 'bold'),
                    ('letter-spacing', '0.05em'),
                ]},
                {'selector': 'tbody tr:hover td', 'props': [
                    ('background-color', '#16213e'),
                ]},
                {'selector': 'td', 'props': [
                    ('padding', '6px 10px'),
                    ('border-color', '#2a2a4a'),
                ]},
            ])
            st.dataframe(
                styled_df,
                hide_index=True,
                use_container_width=True,
            )
            # st.dataframe(st.session_state.data, hide_index=True,
            #              use_container_width=True)


    

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

    if(model_select=='Support Vector Machine'):
        classification_report= support_vector_machine(st.session_state.data, selected_features, selected_target)
        classification_dataframe = pd.DataFrame(classification_report)
        accuracy = classification_dataframe["accuracy"].unique()[0]
        classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
        st.dataframe(classification_dataframe)
        st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

    if(model_select=='Naive Bayes'):
        classification_report= naive_bayes(st.session_state.data, selected_features, selected_target)
        classification_dataframe = pd.DataFrame(classification_report)
        accuracy = classification_dataframe["accuracy"].unique()[0]
        classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
        st.dataframe(classification_dataframe)
        st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))
    
    if(model_select=='K-Nearest Neighbors'):
        classification_report= k_neighbor(st.session_state.data, selected_features, selected_target)
        classification_dataframe = pd.DataFrame(classification_report)
        accuracy = classification_dataframe["accuracy"].unique()[0]
        classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
        st.dataframe(classification_dataframe)
        st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

    if(model_select=='Decision Tree'):
        classification_report= dec_tree(st.session_state.data, selected_features, selected_target)
        classification_dataframe = pd.DataFrame(classification_report)
        accuracy = classification_dataframe["accuracy"].unique()[0]
        classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
        st.dataframe(classification_dataframe)
        st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

    if(model_select=='Linear Regression'):
        metrics= linear_regression(st.session_state.data, selected_features, selected_target)
        st.dataframe(metrics)
        print(metrics)

    if(model_select=='Decision Tree Regression'):
        metrics= dec_tree_reg(st.session_state.data, selected_features, selected_target)
        st.dataframe(metrics)
        print(metrics)

    if(model_select=='Random Forest Regression'):
        metrics= ran_for_reg(st.session_state.data, selected_features, selected_target)
        st.dataframe(metrics)
        print(metrics)

    
        


if st.session_state.logged_in:
    try:
        show_main_platform()
    except:
        pass
else:
    welcome_page()
