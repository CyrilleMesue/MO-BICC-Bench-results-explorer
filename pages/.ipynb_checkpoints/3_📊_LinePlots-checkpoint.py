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

st.sidebar.caption("**Feature Selectors to Include**")
feature_selectors = list(val_data['featureSelector'].unique())
selected_featureSelectors = [st.sidebar.checkbox(method, value = True) for method in feature_selectors]
selected_featureSelectors = [feature_selectors[i] for i, value in enumerate(selected_featureSelectors) if value]

# filter feature selectors
df = val_data.copy()
subset_val_data = df[df['featureSelector'].isin(selected_featureSelectors)][df['modelName']==selected_model_name][df['numFeatures']==selected_num_biomarkers]

# Ensure numeric x for correct ordering
subset_val_data['groundtruth_cutoff'] = pd.to_numeric(subset_val_data['groundtruth_cutoff'], errors='coerce')

validation_sources = pd.unique(subset_val_data['validationsource'])

# Color map for methods (colors)
palette = px.colors.qualitative.Set2
color_map = {m: palette[i % len(palette)] for i, m in enumerate(selected_featureSelectors)}

# Symbol map for validation sources (markers)
available_symbols = ["circle", "square", "diamond", "cross", "triangle-up",
                     "x", "triangle-down", "star", "hexagon", "pentagon"]
symbol_map = {v: available_symbols[i % len(available_symbols)] for i, v in enumerate(validation_sources)}

fig = go.Figure()

# --- Real data traces (no legend) ---
feature_type_map = {'CTD':'CTD:Gene', 'CTD-pathways':'CTD:Pathway', 'GeneCards':'GeneCards:Gene', 'HMDD':'HMDD:miRNA'}
for m in selected_featureSelectors:
    for v in validation_sources:
        sub = subset_val_data[(subset_val_data['featureSelector'] == m) & (subset_val_data['validationsource'] == v)] 
        if sub.empty:
            continue

        model_score = round(sub[model_metric_column].iloc[0], 3)
        fig.add_trace(go.Scatter(
            x=sub['groundtruth_cutoff'],
            y=sub[selected_validation_scorer],
            mode='lines+markers',
            line=dict(color=color_map[m], width=2),
            marker=dict(symbol=symbol_map[v], size=7, line=dict(width=0)),
            showlegend=False,  # hide these from legend
            hovertemplate=(
                "Feature Selector: " + str(m) + "<br>"
                "Validation Source: " + feature_type_map[str(v)] + "<br>"
                "Ground Truth Cutoff: %{x}<br>"
                f"{selected_validation_scorer.upper()}: %{{y}}<br>" 
                f'{selected_model_name} Model {model_metric} Score: {model_score} <extra></extra>'
            )
        ))

# --- Legend proxies ---
# Methods (colors)
for m in selected_featureSelectors:
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=color_map[m], width=3),
        name=str(m),
        legendgroup="Feature Selector",
        legendgrouptitle_text="Feature Selector",
        showlegend=True
    ))

# Validation sources (markers)
for v in validation_sources:
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(symbol=symbol_map[v], size=9, color="black"),
        name=str(v),
        legendgroup="Validation Source",
        legendgrouptitle_text="Validation Source",
        showlegend=True
    ))

# Layout & styling 
score = f"{selected_validation_scorer.upper()} Score" if 'Score' not in selected_validation_scorer.upper() else selected_validation_scorer.upper()
fig.update_layout(
    template="simple_white",
    title=dict(
        text=f"{score} Score Across GroundTruth Cutoffs",
        x=0.5, xanchor="center"
    ),
    xaxis_title="Ground Truth Cutoff",
    yaxis_title=selected_validation_scorer.upper(),
    legend=dict(
        traceorder="grouped",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        title_font=dict(size=10),
        font=dict(size=9)
    ),
    width=900, height=600,
    margin=dict(l=60, r=200, t=70, b=60)
)
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Streamlit render
st.plotly_chart(fig, use_container_width=True)













####################################################################################################
# End
####################################################################################################





