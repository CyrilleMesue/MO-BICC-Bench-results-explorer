import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Page settings (keep this first) ---
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploration Of MyRIAD Results!')

import streamlit as st

# # Set a fixed sidebar width
# st.markdown(
#     """
#     <style>
#         [data-testid="stSidebar"] {
#             min-width: 230px;
#             max-width: 250px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# --- Helpers (cache I/O) ---
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        if path.endswith("val_results.csv"):
            return pd.DataFrame()
        else:
            raise

@st.cache_data
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def dataset_paths(dataset_name: str, experiment_name: str) -> dict:
    base = Path("data") / dataset_name / experiment_name
    return {
        "cv": base / "cross-validation-results.csv",
        "val": base / "val_results.csv",
        "ranks": base / "BiomarkerRanks.csv",
        "featnames": base / "featurenames.json",
    }

# --- UI: dataset picker ---
col1,col2, col3 = st.columns([3,4,5])
experimental_designs = {"ROSMAP":{"Single Omics":['miRNA_data','dna_methylation_data','gene_expression_data'],
                                  "Dual Omics":['miRNA_and_gene_expression_data','miRNA_and_dna_methylation_data','gene_expression_and_dna_methylation_data'],
                                  "Triple Omics":['miRNA_and_gene_expression_and_dna_methylation_data']
                                 },
                        'MayoRNASeq':{"Single Omics":['metabolomics_data','gene_expression_data','proteomics_data'],
                                      "Dual Omics":['gene_expression_and_proteomics_data','metabolomics_and_gene_expression_data','metabolomics_and_proteomics_data'],
                                      "Triple Omics": ['metabolomics_and_gene_expression_and_proteomics_data']
                                     },
                        'BRCA':{"Single Omics":['miRNA_data','dna_methylation_data','gene_expression_data'],
                                  "Dual Omics":['miRNA_and_gene_expression_data','miRNA_and_dna_methylation_data','gene_expression_and_dna_methylation_data'],
                                  "Triple Omics":['miRNA_and_gene_expression_and_dna_methylation_data']
                               }
                       } 

dataset_name = col1.radio('Select Dataset', options=['ROSMAP', 'BRCA', 'MayoRNASeq'], index=0, key = 1)
omics_type_list = list(experimental_designs[dataset_name].keys())

omics_integration_type = col2.radio('Select Omics Type', options=omics_type_list, index=0, key = 2)
experiment_name_list = experimental_designs[dataset_name][omics_integration_type]

experiment_name = col3.radio('Select Experiment', options=experiment_name_list, index=0, key = 3)

st.session_state["dataset_name"] = dataset_name
paths = dataset_paths(dataset_name, experiment_name)

# --- Load data ---
df = load_csv(str(paths["cv"]))
val_data = load_csv(str(paths["val"]))
biomarker_ranks = load_csv(str(paths["ranks"]))
featurenames = load_json(str(paths["featnames"]))

# --- Clean / enrich once ---
# Ensure numFeatures is integer, then optional categorical for grouping/plots
if "numFeatures" in df.columns:
    df["numFeatures"] = pd.to_numeric(df["numFeatures"], errors="coerce").astype("Int64")

# Feature selection flags
df["FeatureSelection"] = np.where(df["featureSelector"].ne("NONE"), "YES", "NO")

# Classify selector type (case-insensitive, vectorized)
df["SelectorType"] = (
    pd.Series(df["featureSelector"], dtype="string")
      .str.lower()
      .str.contains("rank|weight", regex=True)
      .map({True: "Ensemble", False: "Single"})
      .where(df["featureSelector"].ne("NONE"), "NONE")
)

# Column type lists after dtype fixes
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# --- Store only what you'll reuse across callbacks ---
st.session_state["data"] = df
st.session_state["val_data"] = val_data
st.session_state["biomarker_ranks"] = biomarker_ranks
st.session_state["featurenames"] = featurenames
st.session_state["numeric_cols"] = numeric_cols
st.session_state["categorical_cols"] = categorical_cols

# --- Display (consider tabs to avoid clutter) ---
tab1, tab2, tab3 = st.tabs(["Cross-Validation Results", "Biomarker Validation Results", "Biomarker Ranks"])
with tab1: st.dataframe(df, use_container_width=True)
with tab2: st.dataframe(val_data, use_container_width=True)
with tab3: st.dataframe(biomarker_ranks, use_container_width=True)
