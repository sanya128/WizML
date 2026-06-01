import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
from ml_models.log_reg import logistic_regression
from ml_models.lin_reg import linear_regression
from ml_models.ran_for import random_forest_classifier
from ml_models.svm import support_vector_machine
from ml_models.n_bayes import naive_bayes
from ml_models.knn import k_neighbor
from ml_models.dec_tree import dec_tree
from ml_models.dec_tree_reg import dec_tree_reg
from ml_models.ran_for_reg import ran_for_reg
from ml_models.ridge_reg import ridge_reg
from ml_models.lasso_reg import lasso_reg
from ml_models.svr import svr
import streamlit.components.v1 as components
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

import os
import math 

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "y_pred_prob" not in st.session_state:
    st.session_state.y_pred_prob = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "y" not in st.session_state:
    st.session_state.y= None

if "y_pred" not in st.session_state:
    st.session_state.y_pred= None

if "pca_data" not in st.session_state:
    st.session_state.pca_data = None


def welcome_page():
    st.set_page_config(page_title="WizML", layout="centered")

    
    st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }
        
                
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
    st.set_page_config(initial_sidebar_state="expanded")
    st.markdown("""
    <style>
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        [data-testid="stSidebarCollapseButton"] { display: none !important; }
                
        .stApp {background: #080810; color: #b4d4f0; }
        .stSidebar { background: #0a0e18; }
        section[data-testid="stSidebar"] * { color: #b4d4f0 !important; }
        .stSelectbox label { color: #47aaff !important; }
        .stSelectbox > div > div { background: #0d1520 !important; color: #b4d4f0 !important; border-color: #1a3550 !important; }
        
        .stMultiSelect label { color: #47aaff !important; }
        .stMultiSelect > div > div { background: #0d1520 !important; color: #b4d4f0 !important; border-color: #1a3550 !important; }
        /* force all multiselect inner elements */
        [data-testid="stMultiSelect"] > div { background: #0d1520 !important; border-color: #1a3550 !important; }
        [data-testid="stMultiSelect"] div { background: #0d1520 !important; }
        [data-testid="stMultiSelect"] input { background: #0d1520 !important; color: #b4d4f0 !important; }
        /* the outer container box */
        [data-baseweb="select"] { background: #0d1520 !important; overflow: visible !important; }
        [data-baseweb="select"] > div { background: #0d1520 !important; border-color: #1a3550 !important; border-radius: 6px !important; padding: 6px 8px !important; overflow: visible !important; min-height: 42px !important; }
        [data-baseweb="select"] > div > div { overflow: visible !important; flex-wrap: wrap !important; padding: 0 0 !important; margin: 0 !important; align-items: center !important; }
        [data-baseweb="tag"] { margin: 0 0 0 18px !important; }
        [data-testid="stMultiSelect"] { overflow: visible !important; }
        [data-testid="stMultiSelect"] > div { overflow: visible !important; }
        [data-testid="stMultiSelect"] > div > div { overflow: visible !important; padding: 2px 4px !important; }
        [data-testid="stMultiSelect"] > div > div > div { clip: unset !important; overflow: visible !important; padding: 0 !important; }
        
        
        /* selected tags */
        [data-baseweb="tag"] { background: #1a3550 !important; color: #47aaff !important; border: none !important; border-radius: 4px !important; }
        [data-baseweb="tag"] span { color: #47aaff !important; }
        [data-baseweb="tag"] button { color: #47aaff !important; background: transparent !important; }
        /* remove button inside tag */
        [data-baseweb="tag"] [role="presentation"] { color: #47aaff !important; }
        /* dropdown chevron and clear icons */
        [data-testid="stMultiSelect"] svg { color: #47aaff !important; fill: #47aaff !important; }
        /* the results metric table below */
        [data-testid="stDataFrame"] { background: #0d1520 !important; border-color: #1a3550 !important; } 
                
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

    

   
    # old theme
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
            options = ('Auto MPG', 'Concrete Compressive Strength'),
            index=None,
            placeholder="Select"
        )

        model_select=st.sidebar.selectbox(
            label = 'Select the algorithm',
            options = ('Linear Regression','Ridge Regression', 'Lasso Regression', 'Decision Tree Regression', 'Support Vector Regression', 'Random Forest Regression'),
            index=None,
            placeholder="Select"
        )

    tab1, tab2, tab3 , tab4= st.tabs(["Dataset", "Model results","Visualisation","Description"])

    with tab1:
    
        

        if dataset_select is not None:
            if st.session_state.get("last_dataset") != dataset_select:
                st.session_state.loading_dataset = True
                st.session_state.last_dataset = dataset_select
                st.session_state.data = None
                st.rerun()

        if st.session_state.get("loading_dataset"):
            book_html_path = os.path.join(BASE_DIR, "html_pages/book_loader.html")
            with open(book_html_path, "r") as f:
                book_content = f.read()
            st.markdown("<p style='color:#47aaff; font-family:Courier New; letter-spacing:0.1em;'>LOADING DATASET...</p>", unsafe_allow_html=True)
            components.html(book_content, height=220, scrolling=False)

            dataset = st.session_state.last_dataset
            if dataset == "Iris":
                iris = fetch_ucirepo(id=53)
                df = pd.DataFrame(data=iris.data.features)
                df['target'] = iris.data.targets
            elif dataset == "Heart Disease":
                heart_disease = fetch_ucirepo(id=45)
                df = pd.DataFrame(data=heart_disease.data.features)
                df['target'] = heart_disease.data.targets
            elif dataset == "Auto MPG":
                auto_mpg = fetch_ucirepo(id=9)
                df = pd.DataFrame(data=auto_mpg.data.features)
                df['target'] = auto_mpg.data.targets
            elif dataset == "Concrete Compressive Strength":
                concrete = fetch_ucirepo(id=165)
                df = pd.DataFrame(data=concrete.data.features)
                df['target'] = concrete.data.targets

            st.session_state.data = df
            st.session_state.loading_dataset = False
            st.rerun()

        elif st.session_state.data is not None:
            with st.expander(label="View Dataset", expanded=True):
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
                    {'selector': 'th.col_heading', 'props': [
                        ('background-color', '#0f3460'),
                        ('color', '#a8d8ff'),
                        ('font-family', 'Courier New, monospace'),
                        ('font-size', '13px'),
                        ('padding', '8px'),
                        ('font-weight', 'bold'),
                    ]},
                    {'selector': 'th.col_heading.level0', 'props': [
                        ('background-color', '#0f3460'),
                        ('color', '#a8d8ff'),
                    ]},
                    {'selector': 'tbody tr:hover td', 'props': [
                        ('background-color', '#16213e'),
                    ]},
                    {'selector': 'td', 'props': [
                        ('padding', '6px 10px'),
                        ('border-color', '#2a2a4a'),
                    ]},
                ])
                st.dataframe(styled_df, hide_index=True, use_container_width=True, height=200)

        else:
            with st.expander(label="View Dataset", expanded=True):
                st.markdown("<span style='color:#2a4a6a; font-family:Courier New;'>No dataset loaded yet.</span>", unsafe_allow_html=True)

                # st.dataframe(st.session_state.data, hide_index=True,
                #              use_container_width=True)


        

        if st.session_state.data is not None:
            selected_features = st.multiselect(
                "Select your features",
                options=list(st.session_state.data.columns),
            )
            selected_target = st.multiselect(
                "Select your target",
                options=list(st.session_state.data.columns)
            )
        else:
            selected_features = []
            selected_target = []

    with tab2:
        if selected_features and selected_target:
            if(model_select=='Logistic Regression'):
                classification_report,  y_pred_prob, y_test, y, y_pred= logistic_regression(st.session_state.data, selected_features, selected_target)
                st.session_state.y_test = y_test 
                st.session_state.y_pred_prob = y_pred_prob
                st.session_state.y=y
                st.session_state.y_pred =y_pred
                classification_dataframe = pd.DataFrame(classification_report)
                accuracy = classification_dataframe["accuracy"].unique()[0]
                classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
                st.dataframe(classification_dataframe)
                st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

            if(model_select=='Random Forest'):
                classification_report,  y_pred_prob, y_test, y, y_pred= random_forest_classifier(st.session_state.data, selected_features, selected_target)
                st.session_state.y_test = y_test 
                st.session_state.y_pred_prob = y_pred_prob
                st.session_state.y=y
                st.session_state.y_pred=y_pred
                classification_dataframe = pd.DataFrame(classification_report)
                accuracy = classification_dataframe["accuracy"].unique()[0]
                classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
                st.dataframe(classification_dataframe)
                st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

            if(model_select=='Support Vector Machine'):
                classification_report,  y_pred_prob, y_test, y, y_pred= support_vector_machine(st.session_state.data, selected_features, selected_target)
                st.session_state.y_test = y_test 
                st.session_state.y_pred_prob = y_pred_prob
                st.session_state.y=y
                st.session_state.y_pred=y_pred
                classification_dataframe = pd.DataFrame(classification_report)
                accuracy = classification_dataframe["accuracy"].unique()[0]
                classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
                st.dataframe(classification_dataframe)
                st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

            if(model_select=='Naive Bayes'):
                classification_report,  y_pred_prob, y_test, y, y_pred= naive_bayes(st.session_state.data, selected_features, selected_target)
                st.session_state.y_test = y_test 
                st.session_state.y_pred_prob = y_pred_prob
                st.session_state.y=y
                st.session_state.y_pred=y_pred
                classification_dataframe = pd.DataFrame(classification_report)
                accuracy = classification_dataframe["accuracy"].unique()[0]
                classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
                st.dataframe(classification_dataframe)
                st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))
            
            if(model_select=='K-Nearest Neighbors'):
                classification_report,  y_pred_prob, y_test, y, y_pred= k_neighbor(st.session_state.data, selected_features, selected_target)
                st.session_state.y_test = y_test 
                st.session_state.y_pred_prob = y_pred_prob
                st.session_state.y=y
                st.session_state.y_pred=y_pred
                classification_dataframe = pd.DataFrame(classification_report)
                accuracy = classification_dataframe["accuracy"].unique()[0]
                classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
                st.dataframe(classification_dataframe)
                st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

            if(model_select=='Decision Tree'):
                classification_report,  y_pred_prob, y_test, y, y_pred= dec_tree(st.session_state.data, selected_features, selected_target)
                st.session_state.y_test = y_test 
                st.session_state.y_pred_prob = y_pred_prob
                st.session_state.y=y
                st.session_state.y_pred=y_pred
                classification_dataframe = pd.DataFrame(classification_report)
                accuracy = classification_dataframe["accuracy"].unique()[0]
                classification_dataframe.drop(labels=["accuracy"], axis=1, inplace=True)
                st.dataframe(classification_dataframe)
                st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

            if(model_select=='Linear Regression'):
                metrics, pca_data= linear_regression(st.session_state.data, selected_features, selected_target)
                st.session_state.pca_data = pca_data
                st.dataframe(metrics)


            if(model_select=='Decision Tree Regression'):
                metrics, pca_data= dec_tree_reg(st.session_state.data, selected_features, selected_target)
                st.session_state.pca_data = pca_data
                st.dataframe(metrics)


            if(model_select=='Random Forest Regression'):
                metrics, pca_data= ran_for_reg(st.session_state.data, selected_features, selected_target)
                st.session_state.pca_data = pca_data
                st.dataframe(metrics)


            if(model_select=='Ridge Regression'):
                metrics, pca_data= ridge_reg(st.session_state.data, selected_features, selected_target)
                st.session_state.pca_data = pca_data
                st.dataframe(metrics)


            if(model_select=='Lasso Regression'):
                metrics, pca_data= lasso_reg(st.session_state.data, selected_features, selected_target)
                st.session_state.pca_data = pca_data
                st.dataframe(metrics)


            if(model_select=='Support Vector Regression'):
                metrics, pca_data= svr(st.session_state.data, selected_features, selected_target)
                st.session_state.pca_data = pca_data
                st.dataframe(metrics)
                
            
    with tab3:
        if(select_method=='Classification'):
            col1, col2 = st.columns((2,2))
            with col1:
                with st.container(border=True, width="content", height="content"):
                    classes=np.unique(st.session_state.y)
                    y_test_binarized =label_binarize(st.session_state.y_test, classes=classes)
                    n_classes = len(classes)
                    cmap = plt.cm.get_cmap('tab10', n_classes)
                    fig,ax=plt.subplots(figsize=(6,5), facecolor='#080810')
                    ax.set_facecolor('#080810')
                    for class_index in range(len(np.unique(st.session_state.y))):
                        y_pred_class=st.session_state.y_pred_prob[:, class_index]
                        y_test_class=y_test_binarized[:,class_index]
                        fpr,tpr,thresholds=roc_curve(y_true=y_test_class, y_score=y_pred_class)
                        auc_score=roc_auc_score(y_true=y_test_class,y_score=y_pred_class)
                        ax.plot(fpr,tpr, label=f"AUC={auc_score:.2f}",color=cmap(class_index), linewidth=2.5)
                    ax.plot([0,1],[0,1], "--", color="#4a6a8a", label="random", linewidth=1.5)
                    ax.set_xlabel("False Positive Rate", fontsize=11, color='#b4d4f0', fontweight='bold')
                    ax.set_ylabel("True Positive Rate", fontsize=11, color='#b4d4f0', fontweight='bold')
                    ax.set_title("ROC Curve", fontsize=13, color='#b4d4f0', fontweight='bold')
                    ax.tick_params(labelsize=10, colors='#b4d4f0')

                    ax.xaxis.label.set_color('#b4d4f0')
                    ax.yaxis.label.set_color('#b4d4f0')
                    ax.title.set_color('#b4d4f0')

                    for spine in ax.spines.values():
                        spine.set_edgecolor('#1a3550')
                        spine.set_linewidth(1.2)

                    ax.grid(color='#1a3550', linestyle='--', linewidth=0.7, alpha=0.6)
                    ax.legend(fontsize=10, facecolor='#0d1520', edgecolor='#47aaff', labelcolor='#b4d4f0', loc='lower right', framealpha=0.95)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)

            with col2:
                with st.container(border=True, width="content", height="content"):
                    mpl.rcParams.update({
                        'font.size': 10,
                        'axes.titlesize': 12,
                        'axes.labelsize': 11,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'legend.fontsize': 10,
                    })

                    y_test_class_cm = st.session_state.y_test.iloc[:,0].to_numpy().ravel()
                    y_pred_class_cm = st.session_state.y_pred
                    print(y_test_class_cm)
                    print(y_pred_class_cm)

                    cm = confusion_matrix(y_test_class_cm,y_pred_class_cm)
                    class_names = sorted(set(y_test_class_cm) | set(y_pred_class_cm))
                    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#080810')
                    ax.set_facecolor('#080810')
                    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.yaxis.set_tick_params(color='#b4d4f0', labelsize=10)
                    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#b4d4f0')
                    cbar.outline.set_edgecolor('#1a3550')
                    cbar.outline.set_linewidth(1.2)
                    cbar.set_label('Count', color='#b4d4f0', fontsize=11, fontweight='bold')

                    thresh = cm.max() / 2
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                    ha='center', va='center', fontsize=14, fontweight='bold',
                                    color='white' if cm[i, j] > thresh else '#1a1a2e')

                    ax.set_xticks(range(len(class_names)))
                    ax.set_yticks(range(len(class_names)))
                    ax.set_xticklabels(class_names, color='#b4d4f0', fontsize=11, fontweight='bold')
                    ax.set_yticklabels(class_names, color='#b4d4f0', fontsize=11, fontweight='bold')

                    ax.set_xlabel('Predicted Labels', color='#b4d4f0', fontsize=12, fontweight='bold')
                    ax.set_ylabel('True Labels', color='#b4d4f0', fontsize=12, fontweight='bold')
                    ax.set_title('Confusion Matrix', color='#b4d4f0', fontsize=14, fontweight='bold', pad=15)

                    for spine in ax.spines.values():
                        spine.set_edgecolor('#1a3550')
                        spine.set_linewidth(1.2)

                    ax.tick_params(colors='#b4d4f0', labelsize=11)

                    plt.tight_layout()
                    st.pyplot(fig)
                    mpl.rcParams.update(mpl.rcParamsDefault)

        if(select_method=="Regression"):
            col1, col2 = st.columns((2, 2))
            with col1:
                with st.container(border=True, width="content", height="content"):
                    if st.session_state.pca_data is not None:
                        pca_data = st.session_state.pca_data
                        X_test_pca = pca_data['X_test_pca']
                        y_test = pca_data['y_test']
                        bestfit_x = pca_data['bestfit_x']
                        bestfit_line = pca_data['bestfit_line']

                        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#080810')
                        ax.set_facecolor('#080810')

                        ax.scatter(X_test_pca[:, 0], y_test,
                                    color='#47aaff', alpha=0.6, s=50, edgecolors='#47aaff', linewidth=0.5)

                        ax.plot(bestfit_x, bestfit_line, color='#ff6b6b', linewidth=2.5, label='Best Fit Line')

                        
                        corr = np.corrcoef(X_test_pca[:, 0], y_test)[0, 1]
                        ax.set_xlabel('PC1', color='#b4d4f0')
                        ax.set_ylabel('Target', color='#b4d4f0')
                        ax.set_title(f'PC1 vs Target  (r = {corr:.2f})', color='#b4d4f0', fontsize=12)

                        ax.tick_params(colors='#b4d4f0', labelsize=9)
                        ax.xaxis.label.set_color('#b4d4f0')
                        ax.yaxis.label.set_color('#b4d4f0')
                        ax.title.set_color('#b4d4f0')

                        for spine in ax.spines.values():
                            spine.set_edgecolor('#1a3550')

                        ax.grid(color='#1a3550', linestyle='--', linewidth=0.5, alpha=0.5)
                        ax.legend(facecolor='#0d1520', edgecolor='#1a3550', labelcolor='#b4d4f0', fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=False)
                    else:
                        st.markdown("<span style='color:#2a4a6a; font-family:Courier New;'>Run a regression model to see PCA visualization.</span>", unsafe_allow_html=True)

            with col2:
                with st.container(border=True, width="content", height="content"):
                    if st.session_state.pca_data is not None:
                        pca_data = st.session_state.pca_data
                        X_test_pca = pca_data['X_test_pca']
                        residuals = pca_data['residuals']

                        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#080810')
                        ax.set_facecolor('#080810')

                        ax.scatter(X_test_pca[:, 0], residuals, alpha=0.6, s=50,
                                  color='#47aaff', edgecolors='#47aaff', linewidth=0.5)
                        ax.axhline(y=0, color='#ff6b6b', linestyle='--', linewidth=2.5, label='Zero Line')

                        ax.set_xlabel('PC1', color='#b4d4f0', fontsize=10)
                        ax.set_ylabel('Residuals', color='#b4d4f0', fontsize=10)
                        ax.set_title('Residual Plot', color='#b4d4f0', fontsize=12)

                        ax.tick_params(colors='#b4d4f0', labelsize=9)
                        ax.xaxis.label.set_color('#b4d4f0')
                        ax.yaxis.label.set_color('#b4d4f0')

                        for spine in ax.spines.values():
                            spine.set_edgecolor('#1a3550')

                        ax.grid(color='#1a3550', linestyle='--', linewidth=0.5, alpha=0.5, axis='y')
                        ax.legend(facecolor='#0d1520', edgecolor='#1a3550', labelcolor='#b4d4f0', fontsize=8)

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=False)
                    else:
                        st.markdown("<span style='color:#2a4a6a; font-family:Courier New;'>Run a regression model to see residual plot.</span>", unsafe_allow_html=True)


if st.session_state.logged_in:
    try:
        show_main_platform()
    except:
        pass
else:
    welcome_page()
