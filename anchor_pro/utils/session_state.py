import streamlit as st
from pandas import Series, DataFrame
from dataclasses import dataclass
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams
from utils.data_loader import anchor_pro_set_data, anchor_pro_concrete_data

def app_setup():
    """Initialize session state variables"""
    default_data_column = initialize_default_data_column()
    return default_data_column

def initialize_default_data_column() -> list[Series]:
    """Initialize the default data column in session state"""
    if "data_column" not in st.session_state:
        default_data_column: list = []
        design_params = DesignParameters()

        default_series = Series(design_params.combined_dict)
        default_data_column.insert(0, default_series)
        return default_data_column

def save_design_to_session(design_params: DesignParameters):
    """Save current design to session state as a new snapshot (list[Series], newest first)."""
    # counter
    if "data_column_counter" not in st.session_state:
        st.session_state["data_column_counter"] = 1
    st.session_state["data_column_counter"] += 1

    # ensure data_column is a list of Series (recover if it was a single Series)
    existing = st.session_state.get("data_column", [])
    if isinstance(existing, list):
        data_list = existing
    elif isinstance(existing, Series):
        data_list = [existing]  # salvage prior state instead of nuking it
    else:
        data_list = []

    # take an independent snapshot so previous saves don't get mutated later
    snap = Series(design_params.combined_dict).copy(deep=True)
    data_list.insert(0, snap)

    # write back
    st.session_state["data_column"] = data_list

    if st.session_state["data_column_counter"] >= 3:
        st.success("Design saved!")


def get_saved_designs():
    """Get all saved designs from session state"""
    return st.session_state.get("data_column")

def update_active_data_column(data_column_key: str, data):
    """Update precise session state data column value"""
    if st.session_state['data_column']:
        st.session_state['data_column'][0][data_column_key] = data


def update_active_design(design_params: DesignParameters):
    """Update the active design in session state"""
    if "data_column" not in st.session_state or not isinstance(st.session_state["data_column"], list):
        st.session_state["data_column"] = []

    st.session_state['data_column'].insert(0, Series(design_params.combined_dict))

    if len(st.session_state['data_column']) > 1:
        st.session_state['data_column'].pop(1)
