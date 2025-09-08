# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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



####################################################################################################
# Boxplot and Violin Plots
####################################################################################################

st.sidebar.caption("**Select Feature**") 
principal_features = [feature for feature in st.session_state['categorical_cols'] if feature !='numFeatures']
selected_principal_feature = st.sidebar.selectbox("Please select principal feature", principal_features)

st.sidebar.caption("**Select Hue**") 
if selected_principal_feature == 'SelectorType' or selected_principal_feature == 'FeatureSelection':
    hue_columns = [selected_principal_feature]
else:
    hue_columns = ['SelectorType', 'FeatureSelection']
selected_hue_feature = st.sidebar.selectbox("Please select hue feature", hue_columns, index = 0) 


if selected_hue_feature == 'SelectorType':
    data_subset = st.session_state['data'][st.session_state['data']['SelectorType']!= 'NONE']
else:
    data_subset = st.session_state['data'].copy()

st.sidebar.caption("**Select Score**")
selected_scoring_feature = st.sidebar.selectbox("Please select scoring feature from cross validation", st.session_state['numeric_cols'], index = 6)

# get the row with highest MeanF1 per (featureSelector, modelName) 
best_rows = data_subset.loc[
data_subset.groupby(['featureSelector', 'modelName', 'SelectorType'])[selected_scoring_feature].idxmax()
]

# Plotly Express makes grouped boxplots straightforward
fig = px.box(
    best_rows,
    y=selected_scoring_feature,
    x=selected_principal_feature,
    color=selected_hue_feature,   # hue equivalent
    color_discrete_sequence=px.colors.qualitative.Set2,  # same palette
    points="all"  # show individual points; remove if you want pure boxes
)

# Titles & labels
fig.update_layout(
    title=dict(
        text=f"{selected_scoring_feature} Cross Validation Score Distribution by {selected_principal_feature}",
        x=0.4,  # centers the title
        xanchor="center",
        font=dict(size=16)
    ),
    yaxis_title=f"{selected_scoring_feature} Score",
    xaxis_title=selected_principal_feature,
    margin=dict(l=80, r=200, t=80, b=60),  # space for legend
    legend=dict(
        title=selected_hue_feature,
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02   # move legend outside right
    ),
    boxmode="group"
) 
fig.update_traces(width=1.0, jitter=0, pointpos=0)

# Streamlit render
st.plotly_chart(fig, use_container_width=True)

####################################################################################################
# End
####################################################################################################





