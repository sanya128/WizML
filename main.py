import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import io
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
from ml_models.data_cleaning import clean_dataset
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

if "my_tabs" not in st.session_state:
    st.session_state.my_tabs = "Dataset"

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "upload_detected" not in st.session_state:
    st.session_state.upload_detected = None

if "last_upload_name" not in st.session_state:
    st.session_state.last_upload_name = None

def _detect_task(df: pd.DataFrame) -> str:
    col = df["target"] if "target" in df.columns else df.iloc[:, -1]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col = col.dropna()
    if not pd.api.types.is_numeric_dtype(col):
        return "Discrete"
    if pd.api.types.is_float_dtype(col):
        return "Continuous"
    if col.nunique() <= 20:
        return "Discrete"
    return "Continuous"



from upstash_redis import Redis

def get_redis():
    return Redis(
        url=st.secrets["UPSTASH_REDIS_REST_URL"],
        token=st.secrets["UPSTASH_REDIS_REST_TOKEN"],
    )

def increment_visitor_count():
    r = get_redis()
    is_dev = st.secrets.get("DEV_MODE", False)
    if not is_dev:
        return r.incr("wizml_visitor_count")
    return int(r.get("wizml_visitor_count") or 0)

def get_visitor_count():
    r = get_redis()
    count = r.get("wizml_visitor_count")
    return int(count) if count else 0

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    elif n % 10 == 1:
        suffix = "st"
    elif n % 10 == 2:
        suffix = "nd"
    elif n % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"
    return f"{n}{suffix}"


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
        iframe { background: transparent !important; border: none !important; }
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
        increment_visitor_count()
        st.session_state.logged_in = True
        st.rerun()

    
    BASE_DIR = os.path.dirname(__file__)
    html_path = os.path.join(BASE_DIR, "html_pages/wlcm.html")
    count = get_visitor_count()+1
    visitor_text = f"👾 you are the <b>{ordinal(count)}</b> visitor"

    with open(html_path, "r") as f:
        html_content = f.read()

    # inject visitor text into the HTML before the closing body tag
    html_content = html_content.replace(
        "</body>",
        f"""<p style='text-align:center; color:#47aaff; font-family:Courier New;
            font-size:13px; letter-spacing:0.08em; margin-top:8px;'>
            {visitor_text}</p>
        </body>"""
    )

    components.html(html_content, height=600, scrolling=False)
    

def show_main_platform():
    st.set_page_config(initial_sidebar_state="expanded")
    st.markdown(""" 
    <style>
        #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    hide_avatar_css = """
        <style>
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            display: none;
        }
        .stAppDeployButton {
            display: none !important;
        }
        </style>
    """
    st.markdown(hide_avatar_css, unsafe_allow_html=True)

    st.markdown("""
    <style>     
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        [data-testid="stSidebarCollapseButton"] { display: none !important; }
                
        [data-testid="stMainMenuPopover"] { display: none !important; }
        div[class*="viewerBadge"] { display: none !important; }
        #MainMenu { visibility: hidden !important; }
        .stDeployButton { display: none !important; }
        [data-testid="manage-app-button"] { display: none !important; }
                
        [data-testid="appCreatorAvatar"] { display: none !important; }
        [class*="_viewerBadge"] { display: none !important; }
        [class*="_profileContainer"] { display: none !important; }
        [class*="_container_gzau"] { display: none !important; }
                
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
                
        /* force ALL dataframe header cells including first row */
        [data-testid="stDataFrame"] [role="columnheader"] { 
            background-color: #0f3460 !important; 
            color: #a8d8ff !important; 
        }
        [data-testid="stDataFrame"] [role="gridcell"] { 
            background-color: #1a1a2e !important; 
            color: #c8d8f0 !important; 
        }
                
        /* book loader iframe background */
        iframe { background: #080810 !important; }
                
        /* multiselect tags */
        .stMultiSelect span[data-baseweb="tag"] { background: #0d1520 !important; color: #47aaff !important; border: 1px solid #1a3550 !important; }
        /* dropdown options */
        li[role="option"] { background: #0d1520 !important; color: #b4d4f0 !important; }
        li[role="option"]:hover { background: #1a3550 !important; }
                
        /* SUBMIT BUTTON THEME */
        div.stButton > button {
            background-color: #0f3460 !important;
            color: #b4d4f0 !important;
            border: 1px solid #47aaff !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            min-height: 42px !important;
            transition: all 0.25s ease !important;
        }

        div.stButton > button:hover {
            background-color: #12263a !important;
            color: white !important;
            border: 1px solid #6bc1ff !important;
            box-shadow: 0 0 10px rgba(71,170,255,0.45) !important;
        }

        
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

    

    # ── derive flags ──────────────────────────────────────────────────────────
    has_upload = st.session_state.uploaded_df is not None
    # has_preset is defined AFTER the container, once dataset_select exists

    
    select_method = st.sidebar.selectbox(
        label       = 'Select the type of algorithms',
        options     = ('Regression', 'Classification'),
        index       = None,
        placeholder = "Select",
        disabled    = has_upload,
        key         = "select_method_widget",
    )

    if select_method == 'Classification':
        preset_ds_opts = ('Iris', 'Heart Disease')
    elif select_method == 'Regression':
        preset_ds_opts = ('Auto MPG', 'Concrete Compressive Strength')
    else:
        preset_ds_opts = ('Iris', 'Heart Disease', 'Auto MPG', 'Concrete Compressive Strength')

    dataset_select = st.sidebar.selectbox(
        label       = 'Select the dataset',
        options     = preset_ds_opts,
        index       = None,
        placeholder = "Select",
        disabled    = has_upload,
        key         = "dataset_select_widget"
    )

    has_preset = (
        dataset_select is not None
        and st.session_state.data is not None
        and not has_upload
    )

    # ── OR divider ────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style='display:flex;align-items:center;margin:10px 4px;
        color:#47aaff;font-family:Courier New;font-size:13px;'>
        <div style='flex:1;height:1px;background:#1a3550;'></div>
        <span style='margin:0 10px;'>OR</span>
        <div style='flex:1;height:1px;background:#1a3550;'></div>
    </div>
    """, unsafe_allow_html=True)

    # ── file uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset (CSV or Excel)",
        type     = ["csv","xlsx"],
        disabled = has_preset,
        help     = "To upload a dataset, first remove the selected dataset above.",
    )

    if has_preset:
        st.sidebar.markdown(
            "<div style='margin-top:-8px; padding:5px 10px; border-radius:5px;"
            "border:1px solid #2a2a4a; background:#0a0e18;"
            "color:#4a6a8a; font-family:Courier New; font-size:11px;'>"
            "Remove selected dataset above to enable upload."
            "</div>",
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_upload_name:

            file_ext = uploaded_file.name.split(".")[-1].lower()

            if file_ext == "csv":
                raw_df = pd.read_csv(uploaded_file)
            elif file_ext == "xlsx":
                raw_df = pd.read_excel(uploaded_file)

            raw_df = clean_dataset(raw_df)
            if "target" not in raw_df.columns:
                raw_df = raw_df.rename(columns={raw_df.columns[-1]: "target"})
            detected = _detect_task(raw_df)
            st.session_state.uploaded_df      = raw_df
            st.session_state.upload_detected  = detected
            st.session_state.last_upload_name = uploaded_file.name
            st.session_state.data             = raw_df
            st.rerun()
    elif st.session_state.uploaded_df is not None:
        # file removed — reset
        st.session_state.uploaded_df      = None
        st.session_state.upload_detected  = None
        st.session_state.last_upload_name = None
        st.session_state.data             = None
        st.rerun()

    # re-read flag after potential rerun
    has_upload = st.session_state.uploaded_df is not None

    # ── detection badge ───────────────────────────────────────────────────────
    if has_upload and st.session_state.upload_detected:
        detected    = st.session_state.upload_detected
        badge_color = "#44dd88" if detected == "Continuous" else "#ff6b6b"
        badge_icon  = "📈" if detected == "Continuous" else "🔢"
        st.sidebar.markdown(f"""
        <div style='margin:6px 0 4px 0;padding:8px 12px;border-radius:6px;
            border:1px solid {badge_color};background:#0a0e18;
            color:{badge_color};font-family:Courier New;font-size:12px;'>
            {badge_icon}&nbsp;<b>{detected} target detected</b><br>
            <span style='color:#b4d4f0;font-size:11px;'>
                Showing {'regression' if detected == 'Continuous' else 'classification'} algorithms
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── algorithm selector ────────────────────────────────────────────────────
    if has_upload:
        detected = st.session_state.upload_detected
        if detected == "Continuous":
            algo_options  = ('Linear Regression','Ridge Regression','Lasso Regression','Decision Tree Regression','Support Vector Regression','Random Forest Regression')
            select_method = "Regression"
        else:
            algo_options  = ('Logistic Regression','Random Forest','Decision Tree','Support Vector Machine','Naive Bayes','K-Nearest Neighbors')
            select_method = "Classification"
    else:
        if select_method == 'Classification':
            algo_options = ('Logistic Regression','Random Forest','Decision Tree','Support Vector Machine','Naive Bayes','K-Nearest Neighbors')
        elif select_method == 'Regression':
            algo_options = ('Linear Regression','Ridge Regression','Lasso Regression','Decision Tree Regression','Support Vector Regression','Random Forest Regression')
        else:
            algo_options = ()

    model_select = st.sidebar.selectbox(
        label       = 'Select the algorithm',
        options     = algo_options if algo_options else ['—'],
        index       = None,
        placeholder = "Select",
        disabled    = not algo_options,
    )
    if model_select == '—':
        model_select = None

    def switch_to_tab(tab_name):
        st.session_state.my_tabs = tab_name

    def themed_dataframe(df):
        styled_df = df.style.set_properties(**{
            'background-color': '#1a1a2e',
            'color': '#c8d8f0',
            'border-color': '#2a2a4a',
            'font-family': 'Courier New, monospace',
            'font-size': '13px'
        }).set_table_styles([
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', '#0f3460'),
                    ('color', '#a8d8ff'),
                    ('border-color', '#2a2a4a'),
                    ('font-weight', 'bold'),
                    ('padding', '8px'),
                ]
            },
            {
                'selector': 'th.row_heading',
                'props': [
                    ('background-color', '#0f3460'),
                    ('color', '#a8d8ff'),
                    ('border-color', '#2a2a4a'),
                    ('font-weight', 'bold'),
                ]
            },
            {
                'selector': 'th.blank',
                'props': [
                    ('background-color', '#0f3460'),
                    ('border-color', '#2a2a4a'),
                ]
            },
            {
                'selector': 'tbody tr:hover td',
                'props': [
                    ('background-color', '#16213e'),
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '6px 10px'),
                    ('border-color', '#2a2a4a'),
                ]
            },
        ])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    tab1, tab2, tab2b, tab3, tab4, tab5 = st.tabs(["Dataset", "EDA", "Data Cleaning", "Model results", "Visualisation", "Description"], key="my_tabs", on_change="rerun")
    
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

            df = clean_dataset(df)
            st.session_state.data = df
            st.session_state.loading_dataset = False
            st.rerun()

        elif st.session_state.data is not None:
            if has_upload:
                st.markdown(
                    f"<div style='margin-bottom:8px;padding:5px 10px;border-radius:5px;"
                    f"border:1px solid #1a3550;background:#0a0e18;display:inline-block;"
                    f"color:#47aaff;font-family:Courier New;font-size:12px;'>"
                    f"📂 &nbsp;<b>{st.session_state.last_upload_name}</b></div>",
                    unsafe_allow_html=True,
                )
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
                options=list(st.session_state.data.columns[:-1]),
            )
            selected_target = ['target'] if 'target' in st.session_state.data.columns else []
        else:
            selected_features = []
            selected_target   = []

        needs_cleaning = (
            st.session_state.data is not None and (
                st.session_state.data.isnull().sum().sum() > 0 or
                st.session_state.data.select_dtypes(exclude=np.number).drop(columns=["target"], errors="ignore").shape[1] > 0
            )
        )

        flag = not (
            len(selected_features) > 0 and
            model_select is not None and
            len(model_select) > 0
        ) or needs_cleaning

        left_space, center_button, right_space = st.columns([3, 2, 3])

        with center_button:
            submit = st.button(
                label="Submit",
                on_click=switch_to_tab,
                args=("Model results",),
                disabled=flag,
                use_container_width=True,
            )

        if needs_cleaning:
            st.markdown(
                "<div style='margin-top:10px; padding:8px 14px; border-radius:6px;"
                "border:1px solid #ff6b6b; background:#0a0e18; text-align:center;'>"
                "<p style='color:#ff6b6b; font-family:Courier New; font-size:12px; margin:0;'>"
                "⚠️ Dataset requires cleaning before training. "
                "Head to the <b>Data Cleaning</b> tab first."
                "</p></div>",
                unsafe_allow_html=True,
            )
        if st.session_state.data is not None and not needs_cleaning and (not selected_features or model_select is None):
            missing_parts = []
            if model_select is None:
                missing_parts.append("an algorithm")
            if not selected_features:
                missing_parts.append("features")
            message = " and ".join(missing_parts)
            st.markdown(
                f"<div style='margin-top:10px; padding:8px 14px; border-radius:6px;"
                f"border:1px solid #2a4a6a; background:#0a0e18; text-align:center;'>"
                f"<p style='color:#4a8aaa; font-family:Courier New; font-size:12px; margin:0;'>"
                f"Please select {message} to continue."
                f"</p></div>",
                unsafe_allow_html=True,
            )

    with tab2:
        if st.session_state.data is None:
            st.warning("Please load a dataset first.")
        else:

            df = st.session_state.data
            BG = "#080810"
            SIDEBAR = "#0a0e18"
            TEXT = "#b4d4f0"
            ACCENT = "#47aaff"

            options = [
                "Overview",
                "Distribution",
                "Correlation",
                "Target Analysis",
                "Missing Values"
            ]

            selection = st.segmented_control(
                "",
                options,
                selection_mode="single",
                default="Overview",
                label_visibility="collapsed"
            )

            if selection == "Overview":

                rows = df.shape[0]
                cols = df.shape[1]
                missing = df.isnull().sum().sum()
                duplicates = df.duplicated().sum()
                mem_usage = round(df.memory_usage(deep=True).sum() / 1024, 2)

                # --- Metrics row ---
                with st.container(border=True):
                    m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
                    with m1:
                        st.metric("Rows", f"{rows:,}")
                    with m2:
                        st.metric("Columns", cols)
                    with m3:
                        st.metric("Missing", missing)
                    with m4:
                        st.metric("Duplicates", duplicates)
                    with m5:
                        st.metric("Memory Usage", f"{mem_usage} KB")

                # --- Pie chart + Summary stats side by side ---
                left, right = st.columns([5,8], gap="small")

                with left:
                    with st.container(border=True):
                        st.markdown("<p style='font-size:14px; color:#b4d4f0; margin-bottom:6px;'>Data Types</p>", unsafe_allow_html=True)
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        categorical_cols = df.select_dtypes(exclude=np.number).columns

                        fig, ax = plt.subplots(figsize=(6, 4.6), facecolor=BG)
                        ax.set_facecolor(SIDEBAR)
                        ax.pie(
                            [len(numeric_cols), len(categorical_cols)],
                            labels=[
                                f"Numeric\n({len(numeric_cols)})",
                                f"Categorical\n({len(categorical_cols)})"
                            ],
                            autopct="%1.1f%%",
                            wedgeprops=dict(width=0.4),
                            colors=['#47aaff', '#ff6b6b'],
                            textprops={'color': TEXT, 'fontsize': 13, 'weight': 'bold'},
                            pctdistance=0.75,
                            labeldistance=1.25
                        )

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", facecolor=BG)
                        buf.seek(0)
                        st.image(buf, use_container_width=True)
                        plt.close(fig)

                with right:
                    with st.container(border=True):
                        st.markdown("<p style='font-size:14px; color:#b4d4f0; margin-bottom:6px;'>Summary Statistics</p>", unsafe_allow_html=True)
                        summary_stats = df.describe().T
                        st.dataframe(
                            summary_stats,
                            use_container_width=True,
                            height=4 * 35 + 38  # shows exactly 4 rows, rest scrollable
                        )

                # --- Missing values bar chart ---
                with st.container(border=True):
                    st.markdown("<p style='font-size:14px; color:#b4d4f0; margin-bottom:6px;'>Missing Values by Column</p>", unsafe_allow_html=True)
                    missing_data = df.isnull().sum()

                    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
                    ax.set_facecolor(SIDEBAR)
                    missing_data.plot(kind="bar", ax=ax, color=ACCENT)
                    ax.set_ylabel("Missing Count", color=TEXT, fontweight='bold', fontsize=11)
                    ax.set_xlabel("Columns", color=TEXT, fontweight='bold', fontsize=11)
                    ax.tick_params(colors=TEXT, labelsize=10)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(ACCENT)
                        spine.set_alpha(0.3)
                    plt.xticks(rotation=45, color=TEXT, fontsize=10)
                    st.pyplot(fig, use_container_width=True)

                # --- Data preview + Feature summary ---
                col1, col2 = st.columns(2, gap="small")

                with col1:
                    with st.container(border=True):
                        st.markdown("<p style='font-size:14px; color:#b4d4f0; margin-bottom:6px;'>Data Preview</p>", unsafe_allow_html=True)
                        st.dataframe(
                            df.head(5),
                            hide_index=True,
                            use_container_width=True,
                            height=5 * 35 + 38  # 5 rows visible, rest scrollable
                        )

                with col2:
                    with st.container(border=True):
                        st.markdown("<p style='font-size:14px; color:#b4d4f0; margin-bottom:6px;'>Feature Summary</p>", unsafe_allow_html=True)
                        feature_summary = pd.DataFrame({
                            "Feature": df.columns,
                            "Unique Values": [df[col].nunique() for col in df.columns]
                        })
                        st.dataframe(
                            feature_summary,
                            use_container_width=True,
                            height=5 * 35 + 38  # 5 rows visible, rest scrollable
                        )

            # ==================================================
            # DISTRIBUTION
            # ==================================================

            elif selection == "Distribution":

                numeric_cols = list(
                    df.select_dtypes(
                        include=np.number
                    ).columns
                )

                if len(numeric_cols) == 0:

                    st.info("No numerical columns available.")

                else:

                    selected_col = st.selectbox(
                        "Select Feature",
                        numeric_cols
                    )
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Histogram")

                        fig, ax = plt.subplots(
                            figsize=(6, 4),
                            facecolor=BG
                        )
                        ax.set_facecolor(SIDEBAR)

                        ax.hist(
                            df[selected_col].dropna(),
                            bins=20,
                            color=ACCENT,
                            edgecolor=TEXT,
                            alpha=0.8
                        )

                        ax.set_title(
                            f"Distribution of {selected_col}",
                            color=TEXT,
                            fontweight='bold'
                        )
                        ax.set_xlabel("Value", color=TEXT, fontweight='bold')
                        ax.set_ylabel("Frequency", color=TEXT, fontweight='bold')
                        ax.tick_params(colors=TEXT)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(ACCENT)
                            spine.set_alpha(0.3)

                        st.pyplot(fig)

                    with col2:
                        st.subheader("Box Plot")

                        fig, ax = plt.subplots(
                            figsize=(6, 4),
                            facecolor=BG
                        )
                        ax.set_facecolor(SIDEBAR)

                        box = ax.boxplot(
                            df[selected_col].dropna(),
                            patch_artist=True
                        )
                        for patch in box['boxes']:
                            patch.set_facecolor(ACCENT)
                            patch.set_alpha(0.7)
                        for whisker in box['whiskers']:
                            whisker.set_color(TEXT)
                        for cap in box['caps']:
                            cap.set_color(TEXT)

                        ax.set_title(
                            f"Boxplot of {selected_col}",
                            color=TEXT,
                            fontweight='bold'
                        )
                        ax.set_ylabel("Value", color=TEXT, fontweight='bold')
                        ax.tick_params(colors=TEXT)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(ACCENT)
                            spine.set_alpha(0.3)

                        st.pyplot(fig)

                    st.subheader("Statistics")

                    st.dataframe(
                        df[[selected_col]]
                        .describe()
                        .T,
                        use_container_width=True
                    )

            # ==================================================
            # CORRELATION
            # ==================================================

            elif selection == "Correlation":

                numeric_df = df.select_dtypes(
                    include=np.number
                )

                if numeric_df.shape[1] < 2:

                    st.info(
                        "Need at least 2 numerical columns."
                    )

                else:

                    st.subheader(
                        "Correlation Matrix"
                    )

                    corr = numeric_df.corr()

                    fig, ax = plt.subplots(
                        figsize=(10, 7),
                        facecolor=BG
                    )
                    ax.set_facecolor(SIDEBAR)

                    heatmap = ax.imshow(
                        corr,
                        aspect="auto",
                        cmap='coolwarm'
                    )

                    ax.set_xticks(
                        range(len(corr.columns))
                    )

                    ax.set_yticks(
                        range(len(corr.columns))
                    )

                    ax.set_xticklabels(
                        corr.columns,
                        rotation=90,
                        color=TEXT
                    )

                    ax.set_yticklabels(
                        corr.columns,
                        color=TEXT
                    )

                    cbar = plt.colorbar(
                        heatmap,
                        ax=ax
                    )
                    cbar.ax.tick_params(colors=TEXT)

                    st.pyplot(fig)

                    st.subheader(
                        "Correlation Values"
                    )

                    st.dataframe(
                        corr,
                        use_container_width=True
                    )

            # ==================================================
            # TARGET ANALYSIS
            # ==================================================

            elif selection == "Target Analysis":

                if "target" not in df.columns:

                    st.warning(
                        "No target column found."
                    )

                else:

                    target = "target"

                    if pd.api.types.is_numeric_dtype(
                        df[target]
                    ):

                        st.subheader(
                            "Target Distribution"
                        )

                        fig, ax = plt.subplots(
                            figsize=(7, 4),
                            facecolor=BG
                        )
                        ax.set_facecolor(SIDEBAR)

                        ax.hist(
                            df[target].dropna(),
                            bins=20,
                            color=ACCENT,
                            edgecolor=TEXT,
                            alpha=0.8
                        )
                        ax.set_title("Distribution", color=TEXT, fontweight='bold')
                        ax.set_xlabel("Value", color=TEXT, fontweight='bold')
                        ax.set_ylabel("Frequency", color=TEXT, fontweight='bold')
                        ax.tick_params(colors=TEXT)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(ACCENT)
                            spine.set_alpha(0.3)

                        st.pyplot(fig)

                        numeric_cols = [
                            col
                            for col in df.select_dtypes(
                                include=np.number
                            ).columns
                            if col != target
                        ]

                        if len(numeric_cols) > 0:

                            feature = st.selectbox(
                                "Select Feature",
                                numeric_cols
                            )

                            st.subheader(
                                f"{feature} vs Target"
                            )

                            fig, ax = plt.subplots(
                                figsize=(7, 4),
                                facecolor=BG
                            )
                            ax.set_facecolor(SIDEBAR)

                            ax.scatter(
                                df[feature],
                                df[target],
                                color=ACCENT,
                                alpha=0.6,
                                edgecolors=TEXT,
                                linewidth=0.5
                            )

                            ax.set_xlabel(
                                feature,
                                color=TEXT,
                                fontweight='bold'
                            )

                            ax.set_ylabel(
                                target,
                                color=TEXT,
                                fontweight='bold'
                            )
                            ax.tick_params(colors=TEXT)
                            for spine in ax.spines.values():
                                spine.set_edgecolor(ACCENT)
                                spine.set_alpha(0.3)

                            st.pyplot(fig)

                    else:

                        counts = (
                            df[target]
                            .value_counts()
                        )

                        fig, ax = plt.subplots(
                            figsize=(7, 4),
                            facecolor=BG
                        )
                        ax.set_facecolor(SIDEBAR)

                        counts.plot(
                            kind="bar",
                            ax=ax,
                            color=ACCENT,
                            edgecolor=TEXT
                        )
                        ax.set_title("Target Counts", color=TEXT, fontweight='bold')
                        ax.set_ylabel("Count", color=TEXT, fontweight='bold')
                        ax.set_xlabel("Class", color=TEXT, fontweight='bold')
                        ax.tick_params(colors=TEXT)
                        plt.xticks(rotation=45, color=TEXT)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(ACCENT)
                            spine.set_alpha(0.3)

                        st.pyplot(fig)

            # ==================================================
            # MISSING VALUES
            # ==================================================

            elif selection == "Missing Values":

                st.subheader(
                    "Missing Values Analysis"
                )

                missing_data = (
                    df.isnull()
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Summary")
                    st.dataframe(
                        pd.DataFrame({
                            "Column":
                                missing_data.index,
                            "Missing":
                                missing_data.values
                        }),
                        use_container_width=True
                    )

                with col2:
                    st.subheader("Percentage")
                    percent = (
                        df.isnull().mean() * 100
                    ).round(2)

                    st.dataframe(
                        pd.DataFrame({
                            "Column":
                                percent.index,
                            "Missing %":
                                percent.values
                        }),
                        use_container_width=True
                    )

                fig, ax = plt.subplots(
                    figsize=(10, 5),
                    facecolor=BG
                )
                ax.set_facecolor(SIDEBAR)

                missing_data.plot(
                    kind="bar",
                    ax=ax,
                    color=ACCENT,
                    edgecolor=TEXT
                )

                ax.set_ylabel(
                    "Missing Count",
                    color=TEXT,
                    fontweight='bold'
                )
                ax.set_xlabel(
                    "Columns",
                    color=TEXT,
                    fontweight='bold'
                )
                ax.tick_params(colors=TEXT)
                plt.xticks(rotation=45, color=TEXT)
                for spine in ax.spines.values():
                    spine.set_edgecolor(ACCENT)
                    spine.set_alpha(0.3)

                st.pyplot(fig)
            
    with tab2b:
        if st.session_state.data is None:
            st.warning("Please load a dataset first.")
        else:
            df = st.session_state.data.copy()

            BG     = "#080810"
            SIDEBAR= "#0a0e18"
            TEXT   = "#b4d4f0"
            ACCENT = "#47aaff"

            num_cols  = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols  = df.select_dtypes(exclude=np.number).columns.tolist()
            # exclude target from cleaning scope display but keep it in df
            feat_num  = [c for c in num_cols if c != "target"]
            feat_cat  = [c for c in cat_cols if c != "target"]

            missing_total  = df.isnull().sum().sum()
            duplicate_total= df.duplicated().sum()
            constant_cols  = [c for c in df.columns if df[c].nunique() <= 1]
            high_card_cols = [c for c in feat_cat if df[c].nunique() > 20]
            neg_cols       = [c for c in feat_num if (df[c] < 0).any()]

            # ── Dataset Health ────────────────────────────────────────────────
            st.markdown(
                "<p style='color:#47aaff; font-family:Courier New; font-size:15px;"
                "font-weight:bold; letter-spacing:0.08em; margin-bottom:6px;'>"
                "DATASET HEALTH</p>",
                unsafe_allow_html=True,
            )

            health_items = [
                ("Missing Values",       missing_total,           missing_total > 0),
                ("Duplicate Rows",       duplicate_total,         duplicate_total > 0),
                ("Numerical Columns",    len(feat_num),           False),
                ("Categorical Columns",  len(feat_cat),           False),
                ("Constant Columns",     len(constant_cols),      len(constant_cols) > 0),
                ("High Cardinality Cols",len(high_card_cols),     len(high_card_cols) > 0),
                ("Columns w/ Negatives", len(neg_cols),           False),
            ]

            rows_html = ""
            for label, value, warn in health_items:
                color = "#ff6b6b" if warn else TEXT
                rows_html += (
                    f"<tr>"
                    f"<td style='padding:5px 16px; color:{color}; font-family:Courier New; font-size:13px;'>{label}</td>"
                    f"<td style='padding:5px 16px; color:{color}; font-family:Courier New; font-size:13px; font-weight:bold;'>: {value}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"<table style='border:1px solid #1a3550; border-radius:6px; "
                f"background:#0a0e18; width:100%; border-collapse:collapse;'>"
                f"{rows_html}</table>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Cleaning Actions ──────────────────────────────────────────────
            st.markdown(
                "<p style='color:#47aaff; font-family:Courier New; font-size:15px;"
                "font-weight:bold; letter-spacing:0.08em; margin-bottom:6px;'>"
                "CLEANING ACTIONS</p>",
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                do_duplicates = st.checkbox("Remove Duplicate Rows", value=duplicate_total > 0)

            with st.container(border=True):
                do_missing = st.checkbox("Handle Missing Values", value=missing_total > 0)
                if do_missing:
                    missing_strategy = st.radio(
                        "Strategy",
                        options=["Mean", "Median", "Mode", "Drop Rows with Missing"],
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                else:
                    missing_strategy = "Mean"

            with st.container(border=True):
                do_encode = st.checkbox(
                    "Encode Categorical Features",
                    value=len(feat_cat) > 0,
                    disabled=len(feat_cat) == 0,
                )
                if do_encode and feat_cat:
                    encode_strategy = st.radio(
                        "Encoding",
                        options=["Label Encoding", "One-Hot Encoding"],
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                else:
                    encode_strategy = "Label Encoding"

            with st.container(border=True):
                do_outliers = st.checkbox("Remove Outliers (IQR method)", value=False)
                if do_outliers:
                    outlier_cols = st.multiselect(
                        "Apply to columns (leave empty = all numeric)",
                        options=feat_num,
                    )

            with st.container(border=True):
                do_scale = st.checkbox("Scale Numerical Features", value=False)
                if do_scale:
                    scale_strategy = st.radio(
                        "Scaler",
                        options=["StandardScaler", "MinMaxScaler", "RobustScaler"],
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                else:
                    scale_strategy = "StandardScaler"

            with st.container(border=True):
                do_constant = st.checkbox(
                    "Drop Constant Columns",
                    value=len(constant_cols) > 0,
                    disabled=len(constant_cols) == 0,
                )

            with st.container(border=True):
                do_highcard = st.checkbox(
                    "Drop High Cardinality Columns (> 20 unique)",
                    value=False,
                    disabled=len(high_card_cols) == 0,
                )

            with st.container(border=True):
                do_dtype_fix = st.checkbox(
                    "Fix Mixed / Object Columns (coerce to numeric where possible)",
                    value=False,
                )

            with st.container(border=True):
                do_whitespace = st.checkbox(
                    "Strip Whitespace from String Columns",
                    value=len(feat_cat) > 0,
                    disabled=len(feat_cat) == 0,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Apply Button ──────────────────────────────────────────────────
            _, center, _ = st.columns([3, 2, 3])
            with center:
                apply = st.button("Apply Cleaning", use_container_width=True)

            if apply:
                from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

                original_shape = df.shape
                df_clean = df.copy()

                # 1. strip whitespace
                if do_whitespace:
                    for c in feat_cat:
                        if c in df_clean.columns:
                            df_clean[c] = df_clean[c].astype(str).str.strip()

                # 2. fix mixed/object columns
                if do_dtype_fix:
                    for c in df_clean.columns:
                        if df_clean[c].dtype == object:
                            converted = pd.to_numeric(df_clean[c], errors='coerce')
                            if converted.notna().sum() > df_clean[c].notna().sum() * 0.5:
                                df_clean[c] = converted

                # 3. constant columns
                if do_constant:
                    cols_to_drop = [c for c in constant_cols if c != "target"]
                    df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')

                # 4. high cardinality
                if do_highcard:
                    cols_to_drop = [c for c in high_card_cols if c != "target"]
                    df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')

                # 5. duplicates
                if do_duplicates:
                    df_clean.drop_duplicates(inplace=True)

                # 6. missing values
                if do_missing:
                    num_cols_now = [c for c in df_clean.select_dtypes(include=np.number).columns if c != "target"]
                    cat_cols_now = [c for c in df_clean.select_dtypes(exclude=np.number).columns if c != "target"]
                    if missing_strategy == "Drop Rows with Missing":
                        df_clean.dropna(inplace=True)
                    elif missing_strategy == "Mean":
                        df_clean[num_cols_now] = df_clean[num_cols_now].fillna(df_clean[num_cols_now].mean())
                        for c in cat_cols_now:
                            df_clean[c].fillna(df_clean[c].mode()[0] if not df_clean[c].mode().empty else "Unknown", inplace=True)
                    elif missing_strategy == "Median":
                        df_clean[num_cols_now] = df_clean[num_cols_now].fillna(df_clean[num_cols_now].median())
                        for c in cat_cols_now:
                            df_clean[c].fillna(df_clean[c].mode()[0] if not df_clean[c].mode().empty else "Unknown", inplace=True)
                    elif missing_strategy == "Mode":
                        for c in num_cols_now + cat_cols_now:
                            df_clean[c].fillna(df_clean[c].mode()[0] if not df_clean[c].mode().empty else 0, inplace=True)

                # 7. encode categoricals
                if do_encode:
                    cat_cols_now = [c for c in df_clean.select_dtypes(exclude=np.number).columns if c != "target"]
                    if encode_strategy == "Label Encoding":
                        le = LabelEncoder()
                        for c in cat_cols_now:
                            df_clean[c] = le.fit_transform(df_clean[c].astype(str))
                    elif encode_strategy == "One-Hot Encoding":
                        df_clean = pd.get_dummies(df_clean, columns=cat_cols_now)
                        # make sure target stays last
                        if "target" in df_clean.columns:
                            cols = [c for c in df_clean.columns if c != "target"] + ["target"]
                            df_clean = df_clean[cols]

                # 8. outliers
                if do_outliers:
                    cols_for_outliers = outlier_cols if outlier_cols else [c for c in df_clean.select_dtypes(include=np.number).columns if c != "target"]
                    for c in cols_for_outliers:
                        if c in df_clean.columns:
                            Q1  = df_clean[c].quantile(0.25)
                            Q3  = df_clean[c].quantile(0.75)
                            IQR = Q3 - Q1
                            df_clean = df_clean[
                                (df_clean[c] >= Q1 - 1.5 * IQR) &
                                (df_clean[c] <= Q3 + 1.5 * IQR)
                            ]

                # 9. scaling
                if do_scale:
                    num_cols_now = [c for c in df_clean.select_dtypes(include=np.number).columns if c != "target"]
                    if scale_strategy == "StandardScaler":
                        scaler = StandardScaler()
                    elif scale_strategy == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    df_clean[num_cols_now] = scaler.fit_transform(df_clean[num_cols_now])

                # ── save back ─────────────────────────────────────────────────
                st.session_state.data = df_clean
                new_shape = df_clean.shape

                # ── result card ───────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                rows_removed   = original_shape[0] - new_shape[0]
                cols_removed   = original_shape[1] - new_shape[1]
                missing_after  = df_clean.isnull().sum().sum()

                st.markdown(
                    "<p style='color:#44dd88; font-family:Courier New; font-size:15px;"
                    "font-weight:bold; letter-spacing:0.08em; margin-bottom:6px;'>"
                    "✓ CLEANING COMPLETED</p>",
                    unsafe_allow_html=True,
                )

                result_items = [
                    ("Original Shape", f"{original_shape[0]} × {original_shape[1]}"),
                    ("New Shape",      f"{new_shape[0]} × {new_shape[1]}"),
                    ("Rows Removed",   rows_removed),
                    ("Cols Removed",   cols_removed),
                    ("Missing Values", missing_after),
                ]

                rows_html = ""
                for label, value in result_items:
                    color = "#ff6b6b" if (label in ("Missing Values",) and value > 0) else "#44dd88" if label in ("Original Shape","New Shape") else TEXT
                    rows_html += (
                        f"<tr>"
                        f"<td style='padding:5px 16px; color:{TEXT}; font-family:Courier New; font-size:13px;'>{label}</td>"
                        f"<td style='padding:5px 16px; color:{color}; font-family:Courier New; font-size:13px; font-weight:bold;'>: {value}</td>"
                        f"</tr>"
                    )

                st.markdown(
                    f"<table style='border:1px solid #1a3550; border-radius:6px;"
                    f"background:#0a0e18; width:100%; border-collapse:collapse;'>"
                    f"{rows_html}</table>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div style='margin-top:12px; padding:8px 14px; border-radius:6px;"
                    "border:1px solid #47aaff; background:#0a0e18;'>"
                    "<p style='color:#47aaff; font-family:Courier New; font-size:12px; margin:0;'>"
                    "💡 Head to the <b>Dataset</b> tab to reselect your features and run the model on the cleaned data."
                    "</p></div>",
                    unsafe_allow_html=True,
                )




    with tab3:
        if selected_features and selected_target:

            target_col = st.session_state.data["target"]
            if target_col.nunique() <= 1:
                st.markdown(
                    "<div style='margin-top:10px; padding:10px 14px; border-radius:6px;"
                    "border:1px solid #ff6b6b; background:#0a0e18;'>"
                    "<p style='color:#ff6b6b; font-family:Courier New; font-size:13px; margin:0;'>"
                    "⚠️ Target column has only 1 unique value after cleaning. "
                    "The model cannot train on this. Go back to <b>Data Cleaning</b> and "
                    "try a less aggressive cleaning strategy (e.g. avoid removing outliers "
                    "on all columns, or use a different missing value strategy)."
                    "</p></div>",
                    unsafe_allow_html=True,
                )
            else:
           
                if(model_select=='Logistic Regression'):
                    classification_report,  y_pred_prob, y_test, y, y_pred= logistic_regression(st.session_state.data, selected_features, selected_target)
                    st.session_state.y_test = y_test 
                    st.session_state.y_pred_prob = y_pred_prob
                    st.session_state.y=y
                    st.session_state.y_pred =y_pred
                    classification_dataframe = pd.DataFrame(classification_report).transpose()
                    accuracy = classification_dataframe.loc["accuracy","precision"]
                    classification_dataframe.drop(index=["accuracy"], inplace=True)
                    classification_dataframe.rename(columns={
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1-score': 'F1-Score',
                        'support': 'Support'
                    }, inplace=True)
                    classification_dataframe = classification_dataframe.reset_index().rename(columns={'index': 'Class'})
                    themed_dataframe(classification_dataframe)
                    st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

                if(model_select=='Random Forest'):
                    classification_report,  y_pred_prob, y_test, y, y_pred= random_forest_classifier(st.session_state.data, selected_features, selected_target)
                    st.session_state.y_test = y_test 
                    st.session_state.y_pred_prob = y_pred_prob
                    st.session_state.y=y
                    st.session_state.y_pred=y_pred
                    classification_dataframe = pd.DataFrame(classification_report).transpose()
                    accuracy = classification_dataframe.loc["accuracy","precision"]
                    classification_dataframe.drop(index=["accuracy"], inplace=True)
                    classification_dataframe.rename(columns={
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1-score': 'F1-Score',
                        'support': 'Support'
                    }, inplace=True)
                    classification_dataframe = classification_dataframe.reset_index().rename(columns={'index': 'Class'})
                    themed_dataframe(classification_dataframe)
                    st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

                if(model_select=='Support Vector Machine'):
                    classification_report,  y_pred_prob, y_test, y, y_pred= support_vector_machine(st.session_state.data, selected_features, selected_target)
                    st.session_state.y_test = y_test 
                    st.session_state.y_pred_prob = y_pred_prob
                    st.session_state.y=y
                    st.session_state.y_pred=y_pred
                    classification_dataframe = pd.DataFrame(classification_report).transpose()
                    accuracy = classification_dataframe.loc["accuracy","precision"]
                    classification_dataframe.drop(index=["accuracy"], inplace=True)
                    classification_dataframe.rename(columns={
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1-score': 'F1-Score',
                        'support': 'Support'
                    }, inplace=True)
                    classification_dataframe = classification_dataframe.reset_index().rename(columns={'index': 'Class'})
                    themed_dataframe(classification_dataframe)
                    st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

                if(model_select=='Naive Bayes'):
                    classification_report,  y_pred_prob, y_test, y, y_pred= naive_bayes(st.session_state.data, selected_features, selected_target)
                    st.session_state.y_test = y_test 
                    st.session_state.y_pred_prob = y_pred_prob
                    st.session_state.y=y
                    st.session_state.y_pred=y_pred
                    classification_dataframe = pd.DataFrame(classification_report).transpose()
                    accuracy = classification_dataframe.loc["accuracy","precision"]
                    classification_dataframe.drop(index=["accuracy"], inplace=True)
                    classification_dataframe.rename(columns={
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1-score': 'F1-Score',
                        'support': 'Support'
                    }, inplace=True)
                    classification_dataframe = classification_dataframe.reset_index().rename(columns={'index': 'Class'})
                    themed_dataframe(classification_dataframe)
                    st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

                if(model_select=='K-Nearest Neighbors'):
                    classification_report,  y_pred_prob, y_test, y, y_pred= k_neighbor(st.session_state.data, selected_features, selected_target)
                    st.session_state.y_test = y_test 
                    st.session_state.y_pred_prob = y_pred_prob
                    st.session_state.y=y
                    st.session_state.y_pred=y_pred
                    classification_dataframe = pd.DataFrame(classification_report).transpose()
                    accuracy = classification_dataframe.loc["accuracy","precision"]
                    classification_dataframe.drop(index=["accuracy"], inplace=True)
                    classification_dataframe.rename(columns={
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1-score': 'F1-Score',
                        'support': 'Support'
                    }, inplace=True)
                    classification_dataframe = classification_dataframe.reset_index().rename(columns={'index': 'Class'})
                    themed_dataframe(classification_dataframe)
                    st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

                if(model_select=='Decision Tree'):
                    classification_report,  y_pred_prob, y_test, y, y_pred= dec_tree(st.session_state.data, selected_features, selected_target)
                    st.session_state.y_test = y_test 
                    st.session_state.y_pred_prob = y_pred_prob
                    st.session_state.y=y
                    st.session_state.y_pred=y_pred
                    classification_dataframe = pd.DataFrame(classification_report).transpose()
                    accuracy = classification_dataframe.loc["accuracy","precision"]
                    classification_dataframe.drop(index=["accuracy"], inplace=True)
                    classification_dataframe.rename(columns={
                        'precision': 'Precision',
                        'recall': 'Recall',
                        'f1-score': 'F1-Score',
                        'support': 'Support'
                    }, inplace=True)
                    classification_dataframe = classification_dataframe.reset_index().rename(columns={'index': 'Class'})
                    themed_dataframe(classification_dataframe)
                    st.metric(label="Model accuracy", value=np.round(accuracy, decimals=2))

                if(model_select=='Linear Regression'):
                    metrics, pca_data= linear_regression(st.session_state.data, selected_features, selected_target)
                    st.session_state.pca_data = pca_data
                    themed_dataframe(metrics)


                if(model_select=='Decision Tree Regression'):
                    metrics, pca_data= dec_tree_reg(st.session_state.data, selected_features, selected_target)
                    st.session_state.pca_data = pca_data
                    themed_dataframe(metrics)


                if(model_select=='Random Forest Regression'):
                    metrics, pca_data= ran_for_reg(st.session_state.data, selected_features, selected_target)
                    st.session_state.pca_data = pca_data
                    themed_dataframe(metrics)


                if(model_select=='Ridge Regression'):
                    metrics, pca_data= ridge_reg(st.session_state.data, selected_features, selected_target)
                    st.session_state.pca_data = pca_data
                    themed_dataframe(metrics)


                if(model_select=='Lasso Regression'):
                    metrics, pca_data= lasso_reg(st.session_state.data, selected_features, selected_target)
                    st.session_state.pca_data = pca_data
                    themed_dataframe(metrics)


                if(model_select=='Support Vector Regression'):
                    metrics, pca_data= svr(st.session_state.data, selected_features, selected_target)
                    st.session_state.pca_data = pca_data
                    themed_dataframe(metrics)
                    
            
    with tab4:
        if(select_method=='Classification'):
            if (
                "y_test" in st.session_state and
                st.session_state.y_test is not None
                ):
                    col1, col2 = st.columns((2,2))
                    with col1:
                        with st.container(border=True, width="content", height="content"):
                            y_pred_prob = np.array(st.session_state.y_pred_prob)
                            if y_pred_prob.ndim == 1:
                                y_pred_prob = np.column_stack([1 - y_pred_prob, y_pred_prob])

                            classes   = np.unique(st.session_state.y)
                            n_classes = len(classes)
                            cmap      = plt.cm.get_cmap('tab10', n_classes)
                            fig, ax   = plt.subplots(figsize=(6,5), facecolor='#080810')
                            ax.set_facecolor('#080810')

                            if n_classes == 2:
                                fpr, tpr, _ = roc_curve(
                                    y_true=st.session_state.y_test,
                                    y_score=y_pred_prob[:, 1],
                                    pos_label=classes[1]
                                )
                                auc_score = roc_auc_score(
                                    y_true=st.session_state.y_test,
                                    y_score=y_pred_prob[:, 1]
                                )
                                ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}", color=cmap(0), linewidth=2.5)
                            else:
                                y_test_binarized = label_binarize(st.session_state.y_test, classes=classes)
                                for class_index in range(n_classes):
                                    y_pred_class = y_pred_prob[:, class_index]
                                    y_test_class = y_test_binarized[:, class_index]
                                    fpr, tpr, _ = roc_curve(y_true=y_test_class, y_score=y_pred_class)
                                    auc_score   = roc_auc_score(y_true=y_test_class, y_score=y_pred_class)
                                    ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}", color=cmap(class_index), linewidth=2.5)

                            # axes styling — outside any loop
                            ax.plot([0,1],[0,1], "--", color="#4a6a8a", label="random", linewidth=1.5)
                            ax.set_xlabel("False Positive Rate", fontsize=11, color='#b4d4f0', fontweight='bold')
                            ax.set_ylabel("True Positive Rate",  fontsize=11, color='#b4d4f0', fontweight='bold')
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

                            y_test_class_cm = np.array(st.session_state.y_test).ravel()
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
                        ax.set_xlabel('Compressed features', color='#b4d4f0')
                        ax.set_ylabel('Target', color='#b4d4f0')
                        ax.set_title(f'Line of Best Fit', color='#b4d4f0', fontsize=12)

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

    with tab5:
        if(model_select=="Linear Regression"):
            st.subheader(':blue[Linear Regression]')
            st.markdown("Linear Regression is a fundamental supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables. It predicts continuous values by fitting a straight line that best represents the data.")
            st.markdown("###### :blue[Equation of the Best-Fit Line:]")
            st.markdown("For simple linear regression (with one independent variable), the best-fit line is represented by the equation.")
            
            st.markdown("##### $$y = mx + c$$", text_alignment="center", )
            st.markdown('''
                        Where: 
                        - y is the predicted value (dependent variable) 
                        - x is the input (independent variable) 
                        - m is the slope of the line (how much y changes when x changes) 
                        - c is the intercept (the value of y when x = 0)
                        ''')

        if(model_select=="Ridge Regression"):
            st.subheader(':blue[Ridge Regression]')
            st.markdown("Ridge Regression is a version of linear regression that adds an L2 penalty to control large coefficient values. \
                        While Linear Regression only minimizes prediction error, it can become unstable when features are highly correlated. \
                        Ridge solves this by shrinking coefficients making the model more stable and reducing overfitting. ")
            st.markdown("###### :blue[L2-Regularized Objective Function]")
            st.markdown("###### $$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2 $$", text_alignment="center")
            st.markdown(''' 
                        - First term measures prediction error 
                        - Second term penalizes squared coefficient values 
                        - λ determines how strongly coefficients are constrained toward zero (without making them exactly zero)
                        ''')
        if(model_select=="Lasso Regression"):
            st.subheader(':blue[Lasso Regression]')
            st.markdown("Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique with L1 regularization that improves model generalization by adding a penalty. This penalty not only controls overfitting but also performs automatic feature selection by shrinking some coefficients exactly to zero, which is useful for high-dimensional datasets.")
            st.markdown("###### :blue[L1-Regularized Objective Function]")
            st.markdown("###### $$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |w_j|$$", text_alignment="center")
            st.markdown('''
                        - First term measures prediction error
                        - Second term penalizes absolute coefficient values
                        - λ determines how strongly coefficients are constrained
                        ''')
            
        if(model_select=="Decision Tree"):
            st.subheader(':blue[Decision Tree Classifier]')
            st.markdown("Decision Tree Classifier is a supervised machine learning algorithm that categorizes data by recursively splitting it based on feature-driven decision rules. \
                        Each internal node represents a condition on a feature, branches denote the outcomes of those conditions and leaf nodes assign the final class label.")
            st.markdown("##### :blue[Gini Index]")
            st.markdown("Gini Index is used to evaluate the quality of a split by measuring the difference between the impurity of \
                        the parent node and the weighted impurity of the child nodes.")
            st.markdown("$$ Gini = 1 - \sum_{i=1}^{n} p_i^2 $$", text_alignment="center")
            st.markdown("A lower Gini Index indicates a more homogeneous or \
                        pure distribution while a higher Gini Index indicates a more impure distribution. So an attribute with a lower Gini index should be preferred.")

        if(model_select=='Logistic Regression'):
            st.subheader(':blue[Logistic Regression]')
            st.markdown("Logistic Regression is a supervised machine learning algorithm used for classification problems. \
                        Unlike linear regression, which predicts continuous values it predicts the probability that an input belongs to a specific class.\n\n It\
                        is used for binary classification where sigmoid function is used to convert inputs into a probability value between 0 and 1.")
            st.markdown(r'''
                        ##### $$ z = wx + b $$ 
                        ##### $$p(z) = \frac{1}{1 + e^{-z}}$$
                        ''', text_alignment="center")
            st.markdown('''
                        where:
                        - w represents the weights assigned to each input feature
                        - x is the input feature or observation
                        - b is the bias term
                        - p(z) represents the predicted probability 
                        ''')
            
        if(model_select=='Decision Tree Regression'):
            st.subheader(":blue[Decision Tree Regressor]")
            st.markdown("Decision Tree Regressor \
                        is used to predict continuous values such as prices or scores using a tree-like structure. \
                        It splits the data into smaller groups based on simple rules derived from input features, helping reduce prediction errors. \
                        At the end of each branch, called a leaf node, the model outputs a value, i.e., usually the average of that group.")
            st.markdown("##### :blue[Gini Index]")
            st.markdown("Gini Index is used to evaluate the quality of a split by measuring the difference between the impurity of \
                        the parent node and the weighted impurity of the child nodes.")
            st.markdown("$$ Gini = 1 - \sum_{i=1}^{n} p_i^2 $$", text_alignment="center")
            st.markdown("A lower Gini Index indicates a more homogeneous or \
                        pure distribution while a higher Gini Index indicates a more impure distribution. So an attribute with a lower Gini index should be preferred.")



        if(model_select=='Support Vector Regression'):
            st.subheader(":blue[Support Vector Regressor]")
            st.markdown('''
                        Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks.
                        It tries to find the best boundary known as a hyperplane that separates different classes in the data. The main goal of \
                        SVM is to maximize the margin between the two classes.

                        The larger the margin the better the model performs on new and unseen data.

                        Support vectors are the points that lie on or inside \
                        the margin and determine the position of the decision boundary.

                        Only support vectors influence the final solution. Other points have zero \
                        contribution to the optimization.
                        ''')

            # ── Theme colors ──────────────────────────────────────────────────────────────
            BG       = "#080810"
            SIDEBAR  = "#0a0e18"
            TEXT     = "#b4d4f0"
            ACCENT   = "#47aaff"
            
            # ── Data ──────────────────────────────────────────────────────────────────────
            np.random.seed(42)
            X = np.linspace(1, 9, 18)
            y = 2 * np.sin(X) + 0.3 * X + np.random.normal(0, 0.3, len(X))
            
            # SVR fit (manual sine approximation for illustration)
            X_line = np.linspace(0.5, 9.5, 300)
            y_fit  = 2 * np.sin(X_line) + 0.3 * X_line
            eps    = 0.45         # epsilon tube half-width
            
            # ── Plot ──────────────────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(SIDEBAR)
            
            # Epsilon tube
            ax.fill_between(X_line, y_fit - eps, y_fit + eps,
                            color=ACCENT, alpha=0.15, label=f'ε-tube (ε={eps})')
            
            # Upper / lower tube boundaries
            ax.plot(X_line, y_fit + eps, '--', color=ACCENT, lw=1.4, alpha=0.7, label='ε boundary')
            ax.plot(X_line, y_fit - eps, '--', color=ACCENT, lw=1.4, alpha=0.7)
            
            # SVR regression line
            ax.plot(X_line, y_fit, color='#ff6b6b', lw=2.2, label='SVR fit (f(x))')
            
            # Classify points: inside tube vs support vectors (outside)
            inside = np.abs(y - (2 * np.sin(X) + 0.3 * X)) <= eps
            outside = ~inside
            
            # Inside-tube points
            ax.scatter(X[inside], y[inside], color=TEXT, s=45, zorder=5, label='Inside ε-tube')
            
            # Support vectors (outside tube) — highlighted
            ax.scatter(X[outside], y[outside], color='#ffcc44', s=70, zorder=6,
                    edgecolors='white', linewidths=0.8, label='Support Vectors')
            
            # Slack lines from support vectors to tube boundary
            for xi, yi in zip(X[outside], y[outside]):
                y_tube = 2 * np.sin(xi) + 0.3 * xi
                boundary = y_tube + eps if yi > y_tube else y_tube - eps
                ax.plot([xi, xi], [yi, boundary], color='#ffcc44', lw=1, ls=':', zorder=4)
            
            # Annotations
            ax.annotate('SVR fit f(x)', xy=(8.2, y_fit[-20]), xytext=(7, y_fit[-20] + 1.1),
                        color='#ff6b6b', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1))
            
            ax.annotate('ε-insensitive\ntube', xy=(5, y_fit[150] + eps * 0.5),
                        xytext=(3.2, y_fit[150] + 1.5), color=ACCENT, fontsize=8.5,
                        arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1))
            
            ax.annotate('Support Vector\n(slack ξ > 0)', xy=(X[outside][0], y[outside][0]),
                        xytext=(X[outside][0] - 2.2, y[outside][0] - 1.2),
                        color='#ffcc44', fontsize=8.5,
                        arrowprops=dict(arrowstyle='->', color='#ffcc44', lw=1))
            
            # ── Styling ───────────────────────────────────────────────────────────────────
            ax.set_title('Support Vector Regression (SVR)', color=TEXT, fontsize=13, pad=12)
            ax.set_xlabel('X', color=TEXT, fontsize=11)
            ax.set_ylabel('y', color=TEXT, fontsize=11)
            ax.tick_params(colors=TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(ACCENT)
                spine.set_alpha(0.4)
            
            legend = ax.legend(facecolor=SIDEBAR, edgecolor=ACCENT, labelcolor=TEXT,
                            fontsize=8.5, loc='upper left')
            
            plt.tight_layout()
            
            st.pyplot(fig)
 


        if(model_select=='Random Forest Regression'):
            st.subheader(":blue[Random Forest Regressor]")
            st.markdown("Random Forest Regression works using the bagging (Bootstrap Aggregating) technique. \
                        Multiple decision trees are trained on different random subsets of the dataset with replacement to train each tree. \
                        Each tree uses a random subset of features while splitting nodes. \
                        The final prediction is obtained by averaging the predictions from all decision trees.")  
            col1, col2, col3 = st.columns([1, 7, 1])
            with col2:
                st.image("randomforest.png", width=600)

        if(model_select=='Random Forest'):
            st.subheader(":blue[Random Forest Classifier]")
            st.markdown("Random Forest Classifier is widely used for classification tasks because it handles large datasets and handles nonlinear relationships well. \
                        It uses Bootstrap Sampling technique where Random rows are picked to train each tree. rees split the data using the best feature from their \
                        random set. Splitting continues until a stopping rule is met (like max depth) \
                        The final prediction is the one most tree agree on.")
            col1, col2, col3 = st.columns([1, 7, 1])
            with col2:
                st.image("ranforclass.png", width=600)

        if(model_select=='Support Vector Machine'):
            st.subheader(":blue[Support Vector Machine]")
            st.markdown("Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks.\
                         It tries to find the best boundary known as hyperplane that separates different classes in the data. \
                        It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog.\n\n The \
                        The main goal of SVM is to maximize the margin between the two classes.\
                         The larger the margin the better the model performs on new and unseen data.")

            BG     = "#080810"
            SIDEBAR= "#0a0e18"
            TEXT   = "#b4d4f0"
            ACCENT = "#47aaff"
            
            # ── Data: two clusters ─────────────────────────────────────────────────────────
            np.random.seed(7)
            # Negative class (bottom-left)
            neg_x = np.random.normal(2.5, 0.6, 12)
            neg_y = np.random.normal(2.5, 0.6, 12)
            # Positive class (top-right)
            pos_x = np.random.normal(7.0, 0.7, 12)
            pos_y = np.random.normal(6.5, 0.7, 12)
            
            # ── Decision boundary: x2 = -x1 + 9  →  slope=-1, intercept=9 ────────────────
            x_line = np.linspace(0.5, 9.5, 300)
            y_db   = -x_line + 9        # decision boundary
            margin = 1.3
            y_pos  = y_db + margin      # +ve hyperplane
            y_neg  = y_db - margin      # -ve hyperplane
            
            # ── Plot ───────────────────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(SIDEBAR)
            
            # Margin shading
            ax.fill_between(x_line, y_neg, y_pos, color=ACCENT, alpha=0.08)
            
            # Decision boundary (solid red)
            ax.plot(x_line, y_db, color='#ff4d4d', lw=2.2)
            
            # +ve hyperplane (green dashed)
            ax.plot(x_line, y_pos, '--', color='#44dd88', lw=1.8)
            
            # -ve hyperplane (red dashed)
            ax.plot(x_line, y_neg, '--', color='#ff4d4d', lw=1.8)
            
            ax.scatter(neg_x, neg_y, marker='x', color='#ff4d4d', s=60, lw=1.5, zorder=5)
            ax.scatter(pos_x, pos_y, marker='x', color='#44dd88', s=60, lw=1.5, zorder=5)

            
            # ── Support vectors (points closest to boundary) ──────────────────────────────
            sv_neg = [(2.8, 4.5), (2.3, 4.9)]   # near -ve hyperplane
            sv_pos = [(6.0, 4.8), (5.5, 5.5)]   # near +ve hyperplane
            
            for sx, sy in sv_neg:
                ax.scatter(sx, sy, marker='x', color='#ff4d4d', s=80, lw=2, zorder=6)
                ax.add_patch(plt.Circle((sx, sy), 0.18, color='#ff4d4d', fill=False, lw=0.8, zorder=7))
            for sx, sy in sv_pos:
                ax.scatter(sx, sy, marker='x', color='#44dd88', s=80, lw=2, zorder=6)
                ax.add_patch(plt.Circle((sx, sy), 0.18, color='#44dd88', fill=False, lw=0.8, zorder=7))
            # ── Margin arrow ──────────────────────────────────────────────────────────────
            # Draw double-headed arrow between the two hyperplanes at x=3
            xm = 3.2
            ym1 = -xm + 9 - margin
            ym2 = -xm + 9 + margin
            ax.annotate('', xy=(xm - 0.5, ym2 + 0.5), xytext=(xm + 0.5, ym1 - 0.5),
                        arrowprops=dict(arrowstyle='<->', color=TEXT, lw=1.4))
            ax.text(xm-1.8, (ym1+ym2)/2+0.6, 'Maximize\nthe Margin', color=TEXT, fontsize=6, ha='center')
            
            ax.text(8.0, -8.0+9+margin+0.15, '"+ ve Hyperplane"', color='#44dd88', fontsize=5.5, ha='right')
            ax.text(8.0, -8.0+9-margin+0.65, '"- ve Hyperplane"', color='#ff4d4d', fontsize=5.5, ha='right')

            
            # ── Support Vectors label ─────────────────────────────────────────────────────
            ax.annotate('Support\nVectors', xy=(6.0, 4.8), xytext=(7.8, 3.8), color=TEXT, fontsize=6,
            arrowprops=dict(arrowstyle='->', color=TEXT, lw=0.7))
            ax.annotate('', xy=(5.5, 5.5), xytext=(7.6, 4.2), 
                        arrowprops=dict(arrowstyle='->', color=TEXT, lw=1))
            ax.annotate('', xy=(2.8, 4.5), xytext=(7.6, 4.2), 
                        arrowprops=dict(arrowstyle='->', color=TEXT, lw=1))
            ax.annotate('', xy=(2.3, 4.9), xytext=(7.6, 4.2), 
                        arrowprops=dict(arrowstyle='->', color=TEXT, lw=1))
            
            
            # ── Legend ────────────────────────────────────────────────────────────────────
            ax.scatter([], [], marker='x', color='#ff4d4d', s=80, lw=2, label='- ve')
            ax.scatter([], [], marker='x', color='#44dd88', s=80, lw=2, label='+ ve')
            ax.legend(facecolor=SIDEBAR, edgecolor=ACCENT, labelcolor=TEXT, fontsize=6, loc='upper right')

            
            # ── Axes styling ──────────────────────────────────────────────────────────────
            ax.set_xlim(0, 9.8)
            ax.set_ylim(0, 9.8)
            ax.set_xlabel('X₁', color=TEXT, fontsize=8)
            ax.set_ylabel('X₂', color=TEXT, fontsize=8)
            ax.tick_params(colors=TEXT, labelsize=6)
            ax.spines['left'].set_edgecolor(TEXT)
            ax.spines['bottom'].set_edgecolor(TEXT)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)


        if(model_select=='K-Nearest Neighbors'):
            st.subheader(":blue[K-Nearest Neighbors]")
            st.markdown(r'''
                        K‑Nearest Neighbor (KNN) is a simple and widely used machine learning technique for classification and regression tasks. 
                        It works by identifying the K closest data points to a given input and making predictions based on the majority class or average value of those neighbors.

                        KNN uses distance metrics to identify nearest neighbor, these neighbors are used for classification and regression task. 
                        
                        To identify nearest neighbor we use below distance metric:

                        1. Euclidean distance:
                        
                        $$ d(x, X_i) = \sqrt{\sum_{j=1}^n (x_j - X_{ij})^2} $$

                        2. Manhattan Distance

                        $$ d(x,y) = \sum_{i=1}^n |x_i - y_i| $$

                        3. Minkowski Distance
                        
                        $$ d(x,y) = \left( \sum_{i=1}^n (x_i - y_i)^p \right)^{1/p} $$
                        ''')

        if(model_select=='Naive Bayes'):
            st.subheader(":blue[Naive Bayes]")
            st.markdown("Naive Bayes is a machine learning classification algorithm that predicts the category of a data point using probability. \
                        It assumes that all features are independent of each other. \
                        Naive Bayes performs well in many real-world applications \
                        such as spam filtering, document categorisation and sentiment analysis.")
            st.markdown(r"$$ P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)} $$", text_alignment="center")
            st.markdown('''
                        Where:

                        - P(y|X): Posterior probability, probability of class y given features X
                        - P(X∣y): Likelihood, probability of features X given class y
                        - P(y): Prior probability of class y
                        - P(X): Marginal likelihood or evidence
                        ''')
            st.image("naivebayes.png", caption="Decision Boundary", width= 600)

if st.session_state.logged_in:
    show_main_platform()

    
else:
    welcome_page()
