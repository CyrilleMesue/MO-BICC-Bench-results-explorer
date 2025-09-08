# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle

####################################################################################################
# Page settings
####################################################################################################
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploration Of MyRIAD Results!')
####################################################################################################
# End
####################################################################################################
if "dataset_name" not in st.session_state:
    st.warning("⚠️ Required data not found. Redirecting to Home page...")
    st.switch_page("🏠_Home.py") 


val_data = st.session_state["val_data"].copy()

models = val_data['modelName'].unique()
selected_model_name = st.sidebar.selectbox("Please select ML Model", models, index = 0)

selected_num_biomarkers = st.sidebar.select_slider("Please select number of biomakers to validate", range(10,101, 10), key = 10)
selected_top_N_val = st.sidebar.select_slider("Please select top N of validation set", range(50,1001, 50), key = 50)

st.sidebar.caption("**Evaluation Metrics**") 
metric_map = {metric.split('Model')[-1]: metric for metric in val_data.columns if 'Model' in metric}
# Modify Precision and Recall labels
# Rebuild with modified keys
new_metric_map = {}
for key, value in metric_map.items():
    if key == "Precision":
        new_metric_map["Precision (PPV)"] = value
    elif key == "Recall":
        new_metric_map["Recall (Sensitivity)"] = value
    else:
        new_metric_map[key] = value

metric_map = new_metric_map
model_metric = st.sidebar.selectbox("Select Model Evaluation Metric", metric_map, index = 3) 

validation_scoring_metrics = ['meanScore', 'medianScore', 'precision', 'recall','f1', 'TP', 'FP', 'FN']
selected_validation_scorer = st.sidebar.selectbox("Please select validation scoring feature", validation_scoring_metrics, index = 5)
model_metric_column = metric_map[model_metric]

# filter feature selectors
df = val_data.copy()
subset_val_data = df[df['modelName']==selected_model_name][df['numFeatures']==selected_num_biomarkers][df['groundtruth_cutoff']==selected_top_N_val]

    
# Plot 
# --- build faceted bar figure --- 
score = f"{selected_validation_scorer.upper()} Score" if 'Score' not in selected_validation_scorer else selected_validation_scorer.upper() 

# --- aggregate like seaborn.barplot (mean, no CI) ---  
df_plot = subset_val_data.copy()

# Order methods (featureSelector) numerically if possible; else lexical
methods = df_plot['featureSelector'].unique().tolist()
try:
    order_methods = sorted(methods, key=lambda x: float(x))
except Exception:
    order_methods = sorted(methods)

sources = df_plot['validationsource'].unique().tolist()
n_sources = len(sources)

fig = px.bar(
    df_plot,
    x='featureSelector',
    y=selected_validation_scorer,
    color='featureSelector',  # color by method, like seaborn palette per category
    facet_row='validationsource',
    category_orders={'featureSelector': order_methods, 'validationsource': sources},
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        'featureSelector': 'Feature Selector',
        selected_validation_scorer: score,
        'validationsource': 'Validation Source'
    },
    template='simple_white',
    title=f'{score} Between Top {selected_num_biomarkers} Biomarker Set And Top {selected_top_N_val} Markers From Validation Sets',
    hover_data={
        'featureSelector': True,
        'validationsource': True,
        selected_validation_scorer: True ,  # format numeric to 3 decimals 
        'groundtruth_cutoff': True,
        model_metric_column: True
    }
)

# Gridlines & styling
fig.update_traces(marker_line_width=0)
fig.update_xaxes(showgrid=True, tickangle=45)
fig.update_yaxes(showgrid=True, title_text=score)

# Replace default facet labels with your custom titles
for ann in fig.layout.annotations:
    # Plotly facet annotation looks like "validationsource=XYZ"
    if ann.text and ann.text.startswith("validationsource="):
        src = ann.text.split("=", 1)[1]
        ann.text = f"{score} per Feature Selector — {src}"
        ann.font.size = 14

# Size scales with number of facet rows (roughly 320px each)
fig.update_layout(
    height=320 * max(n_sources, 1),
    title=dict(x=0.45, xanchor='center'),
    legend=dict(
        title="Feature Selector",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        font=dict(size=9),
        title_font=dict(size=10)
    ),
    margin=dict(l=60, r=200, t=70, b=60)
)

# Streamlit render
st.plotly_chart(fig, use_container_width=True)




####################################################################################################
# End
####################################################################################################





