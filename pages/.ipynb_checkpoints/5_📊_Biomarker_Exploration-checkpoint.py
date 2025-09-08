# --- imports ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.parse

# --- page ---
st.set_page_config(layout="wide")
st.title(':sparkles: Dynamic Exploration Of MyRIAD Results!')

if "dataset_name" not in st.session_state:
    st.warning("⚠️ Required data not found. Redirecting to Home page...")
    st.switch_page("🏠_Home.py") 

# --- helpers ---
def urlq(s: str) -> str:
    return urllib.parse.quote_plus(str(s))

def get_refmet_link(name: str) -> str:
    return f"https://www.metabolomicsworkbench.org/search/sitesearch.php?Name={urlq(name)}"

def get_hmdb_search_link(name: str) -> str:
    base = "https://hmdb.ca/unearth/q"
    params = {"utf8": "✓", "query": name, "searcher": "metabolites", "button": ""}
    return f"{base}?{urllib.parse.urlencode(params)}"

# --- data from session (guard) ---
if not all(k in st.session_state for k in ["biomarker_ranks","dataset_name","featurenames"]):
    st.error("Required session data not found.")
    st.stop()

biomarker_ranks = st.session_state["biomarker_ranks"].copy()
dataset_name = st.session_state["dataset_name"]

# --- omics options by dataset ---
omics_map = {
    "ROSMAP": ['MicroRNA','Genes','Methylation Sites'],
    "BRCA":   ['MicroRNA','Genes','Methylation Sites'],
    "MayoRNASeq": ['Metabolites','Proteins','Genes'],
}
omicstypes = omics_map.get(dataset_name, ['Genes'])

# --- selections (safe index) ---
cols = list(biomarker_ranks.columns)
sel_idx = min(15, len(cols)-1) if cols else 0
feature_selector = st.sidebar.selectbox('Select Feature Selector', options=cols, index=max(sel_idx,0))
omicstype = st.sidebar.radio('Select Omics Level', options=omicstypes, index=0)

# --- subset & ranks ---
subset_data = biomarker_ranks[[feature_selector]].rename(columns={feature_selector: "feature"}).copy()
subset_data['multiomics-level-rank'] = np.arange(len(subset_data)) + 1

# map omicstype -> N key in featurenames
n_map = {
    'MicroRNA': '1', 'Metabolites': '1',
    'Genes': '2', 'Proteins': '2',
    'Methylation Sites': '3'
}
N_key = n_map.get(omicstype)
valid_names = st.session_state["featurenames"].get(N_key, [])

subset_data_by_feature = (
    subset_data[subset_data["feature"].isin(valid_names)]
      .assign(feature=lambda df: df["feature"].str.replace("_y","", regex=False))
      .reset_index(drop=True)
)

num_features = len(subset_data_by_feature)
subset_data_by_feature['omics-level-rank'] = np.arange(num_features) + 1
subset_data_by_feature = subset_data_by_feature.rename(columns={"feature": omicstype})

# --- per-omics links + column_config ---
links = {}
column_config = {}

if omicstype == 'MicroRNA':
    links["mirBASE"] = subset_data_by_feature[omicstype].map(lambda x: f"https://www.mirbase.org/results/?query={urlq(x)}")
    column_config["mirBASE"] = st.column_config.LinkColumn("mirBASE", display_text="Explore microRNA")

elif omicstype == 'Methylation Sites':
    # cg IDs; “Probe Id” has a space → encode term param
    links["EWAS-ATLAS"] = subset_data_by_feature[omicstype].map(
        lambda x: f"https://ngdc.cncb.ac.cn/ewas/search?item={urlq(x)}&term={urlq('Probe Id')}"
    )
    column_config["EWAS-ATLAS"] = st.column_config.LinkColumn("EWAS-ATLAS", display_text="Explore methylation site")

elif omicstype == 'Genes':
    links["GeneCards"] = subset_data_by_feature[omicstype].map(lambda x: f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={urlq(x)}")
    column_config["GeneCards"] = st.column_config.LinkColumn("GeneCards", display_text="Explore gene")

elif omicstype == 'Metabolites':
    links["HMDB"] = subset_data_by_feature[omicstype].map(get_hmdb_search_link)
    links["MetabolomicsWorkbench"] = subset_data_by_feature[omicstype].map(get_refmet_link)
    column_config["HMDB"] = st.column_config.LinkColumn("HMDB", display_text="HMDB entry")
    column_config["MetabolomicsWorkbench"] = st.column_config.LinkColumn("MetabolomicsWorkbench", display_text="RefMet entry")

elif omicstype == 'Proteins':
    # if you pass UniProt IDs or gene symbols; encode either way
    links["UniProtKB"] = subset_data_by_feature[omicstype].map(lambda x: f"https://www.uniprot.org/uniprotkb?query={urlq(x)}")
    column_config["UniProtKB"] = st.column_config.LinkColumn("UniProtKB", display_text="Explore protein")

# attach links (if any)
if links:
    for k,v in links.items():
        subset_data_by_feature[k] = v

# --- render once ---
st.dataframe(
    subset_data_by_feature,
    column_config=column_config,
    use_container_width=True,
    hide_index=True,
)
