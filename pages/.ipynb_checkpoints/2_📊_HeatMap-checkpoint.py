# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go
import numpy as np

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


def fmt_val(v):
    if pd.isna(v):
        return ""                        # or "NA"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"               # keep ints as-is
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.2f}"         # floats to 2 decimals
    # fallback for strings/mixed: try numeric, else keep as string
    try:
        f = float(v)
        return f"{f:.2f}" if not f.is_integer() else f"{int(f)}"
    except Exception:
        return str(v)

        
    
# fetch data here
data = st.session_state["data"].copy() 

show_validation = st.sidebar.toggle('Show Validation Results', value=False)

if not show_validation:  
    all_principal_values = data['featureSelector'].unique()
    selected_principal_values = st.sidebar.multiselect('Select Feature Selectors to include in Plot', 
                                                       all_principal_values, 
                                                       default=all_principal_values)
    
    secondary_columns = ['modelName', 'numFeatures']
    selected_secondary_feature = st.sidebar.selectbox("Select X-Axis Feature", secondary_columns, index = 0) 
    
    all_secondary_values = data[selected_secondary_feature].unique()

    if selected_secondary_feature == 'numFeatures' and 'NONE' in selected_principal_values:
        selected_principal_values.remove('NONE')
        
    subset_data = data[data['featureSelector'].isin(selected_principal_values)].copy()

    
    st.sidebar.caption("**Select Score**") 
    metric_map = {metric.split('Mean')[-1]: metric for metric in st.session_state['numeric_cols'] if 'Mean' in metric}
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
            
    selected_scoring_feature = st.sidebar.selectbox("Please select scoring feature", metric_map, index = 3) 
    selected_scoring_feature = metric_map[selected_scoring_feature]
    
    st.sidebar.caption("**Criteria**")
    selected_agg_criteria = st.sidebar.selectbox("Please select score aggregation criteria across experiments", ['mean', 'max', 'median'], index = 1)  

    # --- aggregate exactly as before ---
    agg_df = (
        subset_data
        .groupby(['featureSelector', selected_secondary_feature])
        .agg({selected_scoring_feature: selected_agg_criteria})
        .reset_index()
    )  

    if selected_secondary_feature == 'numFeatures':  
        agg_df['numFeatures'] = pd.to_numeric(agg_df['numFeatures'].astype(str).str.strip(), errors="coerce") 
        agg_df.sort_values(by=selected_secondary_feature , inplace = True) 
        agg_df[selected_secondary_feature] = agg_df[selected_secondary_feature].apply(lambda x: f'K: {str(x)}')

    x_features_order = list(agg_df[selected_secondary_feature].unique()) 
    
    # --- pivot to matrix ---
    heatmap_data = agg_df.pivot(
        index='featureSelector',
        columns=selected_secondary_feature,
        values=selected_scoring_feature
    ) 
    heatmap_data = heatmap_data[x_features_order]
    # z values and text annotations ("%.3f")
    z = heatmap_data.values.astype(float)
    text = np.empty_like(z, dtype=object)
    text[:] = ""
    mask = ~np.isnan(z)
    text[mask] = np.vectorize(lambda v: f"{v:.3f}")(z[mask])
    
    # titles
    cbar_title = f"{selected_agg_criteria.capitalize()} {selected_scoring_feature.split('Mean')[-1]}"
    title_main = f"{cbar_title} Score: {'featureSelector'} vs {selected_secondary_feature}"
    
    # build figure
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heatmap_data.columns.astype(str),
            y=heatmap_data.index.astype(str),
            colorscale="Viridis",
            colorbar=dict(title=cbar_title),  
            text=text,
            texttemplate="%{text}",   # numeric annotations in cells
            textfont=dict(size=12),        # <-- cell annotation font size
            hovertemplate=(
                f"<b>{'featureSelector'}</b>: %{{y}}<br>"
                f"<b>{selected_secondary_feature}</b>: %{{x}}<br>"
                f"<b>{selected_scoring_feature}</b>: %{{z:.3f}}<extra></extra>"
            ),
            xgap=1, ygap=1  # subtle grid like seaborn linewidths
        )
    )

    # how tall you want each heatmap row (in pixels)
    CELL_H = 24          # try 28–40 depending on your labels
    YGAP_PX = 1          # gap between cells, in pixels (must match your trace's ygap)
    TOP, BOTTOM = 80, 60 # top/bottom margins
    
    # ensure the trace uses the same gap
    fig.update_traces(ygap=YGAP_PX)
    
    rows = heatmap_data.shape[0]
    height = TOP + BOTTOM + rows * CELL_H + max(rows - 1, 0) * YGAP_PX

    fig.update_layout(
        title=dict(
        text=title_main,
        x=0.5,  # centers the title
        xanchor="center",
        font=dict(size=16)
    ),
        xaxis_title=selected_secondary_feature,
        yaxis_title='featureSelector', 
        height=height,
        margin=dict(l=80, r=80, t=TOP, b=BOTTOM),
    )
    
    # match seaborn orientation & ticks
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(autorange="reversed")  # seaborn shows first index at top
    
    # Streamlit render
    st.plotly_chart(fig, use_container_width=True)


else:
    val_data = st.session_state["val_data"].copy()  
    
    all_principal_values = val_data['featureSelector'].unique()
    selected_principal_values = st.sidebar.multiselect('Select Feature Selectors to include in Plot', 
                                                       all_principal_values, 
                                                       default=all_principal_values)
    
    secondary_columns = ['modelName', 'numFeatures']
    selected_secondary_feature = st.sidebar.selectbox("Select X-Axis Feature", secondary_columns, index = 0) 
    
    all_secondary_values = val_data[selected_secondary_feature].unique()

    if selected_secondary_feature == 'numFeatures' and 'NONE' in selected_principal_values:
        selected_principal_values.remove('NONE')
        
    subset_data = val_data[val_data['featureSelector'].isin(selected_principal_values)].copy()

    
    st.sidebar.caption("**Select Score**") 
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
            
    selected_model_scoring_feature = st.sidebar.selectbox("Please select model scoring feature", metric_map, index = 3)  

    selected_scoring_feature = metric_map[selected_model_scoring_feature]

    st.sidebar.caption("**Validation Options**") 
    standby_subset_data = subset_data.copy()
    validation_scoring_metrics = ['meanScore', 'medianScore', 'precision', 'recall','f1', 'TP', 'FP', 'FN']
    selected_validation_scorer = st.sidebar.selectbox("Please select validation scoring feature", validation_scoring_metrics, index = 5) 
    validation_sources =  val_data.validationsource.unique()
    selected_validation_source = st.sidebar.selectbox("Please select validation source", validation_sources, index = 0)   


    if selected_secondary_feature == 'modelName': 
        tertiary_column = 'numFeatures'
        selected_tertiary_feature = st.sidebar.select_slider(
    "Please select number of features to validate",
    options=range(10,101,10), value = 30)   
    else:  
        tertiary_column = 'modelName'
        all_tertiary_values = val_data[tertiary_column].unique()
        selected_tertiary_feature = st.sidebar.selectbox(
    "Please select number of features to validate",
    options=all_tertiary_values, index=0)

    selected_groundtruth_cutoff = st.sidebar.select_slider(
        "Please select groundtruth cutoff",
        options=range(50,1001,50), value = 100)

    subset_data  = standby_subset_data[standby_subset_data[tertiary_column]==selected_tertiary_feature][standby_subset_data['groundtruth_cutoff']==selected_groundtruth_cutoff][standby_subset_data['validationsource']==selected_validation_source] 
    
    st.sidebar.caption("**Criteria**")
    selected_agg_criteria = st.sidebar.selectbox("Please select score aggregation criteria", ['mean', 'max', 'median'], index = 1)  
    
    # --- same aggregation as you already have --- 
    agg_df = (
        subset_data
        .groupby(['featureSelector', selected_secondary_feature])
        .agg({
            selected_scoring_feature: selected_agg_criteria,
            selected_validation_scorer: selected_agg_criteria
        })
        .reset_index()
    )

    agg_df["Scores"] = agg_df.apply(
    lambda row: f"{row[selected_scoring_feature]:.2f}|{fmt_val(row[selected_validation_scorer])}",
    axis=1
    )     
    
    if selected_secondary_feature == 'numFeatures':  
        agg_df['numFeatures'] = pd.to_numeric(agg_df['numFeatures'].astype(str).str.strip(), errors="coerce") 
        agg_df.sort_values(by=selected_secondary_feature , inplace = True) 
        agg_df[selected_secondary_feature] = agg_df[selected_secondary_feature].apply(lambda x: f'K: {str(x)}')

    x_features_order = list(agg_df[selected_secondary_feature].unique())
    # --- pivots ---
    heatmap_data = agg_df.pivot(
        index='featureSelector',
        columns=selected_secondary_feature,
        values=selected_scoring_feature
    )

    heatmap_data = heatmap_data[x_features_order]
    annotation_data = agg_df.pivot(
        index='featureSelector',
        columns=selected_secondary_feature,
        values='Scores'
    ).reindex(index=heatmap_data.index, columns=heatmap_data.columns)
    
    # Optional: ensure numeric array (and mask NaNs for text)
    z = heatmap_data.values.astype(float)
    text = annotation_data.values.astype(str)
    text = np.where(np.isnan(z), "", text)  # hide text where z is NaN
    
    # --- figure --- 
    feature_type_map = {'CTD':'CTD:Gene', 'CTD-pathways':'CTD:Pathway', 'GeneCards':'GeneCards:Gene', 'HMDD':'HMDD:miRNA', 'EWAS-ATLAS':'EWAS-ATLAS:DNA-Methylation'}
    title_main = f"{selected_agg_criteria.capitalize()} {selected_scoring_feature.split('Mean')[-1]} Score " \
                 f"| {feature_type_map[selected_validation_source]}Marker Validation Score ({selected_validation_scorer.upper()}): " \
                 f"{'featureSelector'} vs {selected_secondary_feature}"
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heatmap_data.columns.astype(str),
            y=heatmap_data.index.astype(str),
            colorscale="Viridis",
            colorbar=dict(title=f"{selected_agg_criteria.capitalize()} {selected_scoring_feature.split('Mean')[-1]}"),
            text=text,
            texttemplate="%{text}",   # show "score|val" inside each cell 
            textfont=dict(size=12),
            hovertemplate=(
        f"<b>{'featureSelector'}</b>: %{{y}}<br>"
        f"<b>{selected_secondary_feature}</b>: %{{x}}<br>"
        f"<b>{selected_scoring_feature}</b>: %{{z:.3f}}<br>"
        "<b>Score|Val</b>: %{text}<extra></extra>"
    )
    ,
            xgap=1,  # thin grid lines between cells
            ygap=1
        )
    )


    # how tall you want each heatmap row (in pixels)
    CELL_H = 24          # try 28–40 depending on your labels
    YGAP_PX = 1          # gap between cells, in pixels (must match your trace's ygap)
    TOP, BOTTOM = 80, 60 # top/bottom margins
    
    # ensure the trace uses the same gap
    fig.update_traces(ygap=YGAP_PX) 


    
    rows = heatmap_data.shape[0]
    height = TOP + BOTTOM + rows * CELL_H + max(rows - 1, 0) * YGAP_PX
    # --- styling (map your seaborn font_scale loosely) ---
    fig.update_layout(
        title=dict(
        text=title_main,
        x=0.5,  # centers the title
        xanchor="center",
        font=dict(size=16)
    ),
        xaxis_title=selected_secondary_feature,
        yaxis_title='featureSelector',
        height=height,
        margin=dict(l=80, r=80, t=TOP, b=BOTTOM),
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(autorange="reversed")  # matches seaborn orientation
    
    # Streamlit render
    st.plotly_chart(fig, use_container_width=True)  







