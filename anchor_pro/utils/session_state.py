import streamlit as st
from pandas import Series
from models.design_data import DesignParameters

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "data_column_counter": 0,
        "data_column": [],
        # Probably don't need to initizlize these at start
        # "current_design": None,
        # "anchor_data": None
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def save_design_to_session(design_params: DesignParameters):
    """Save current design to session state"""
    counter: int = st.session_state.get('data_column_counter', 0)
    st.session_state["data_column_counter"] = counter + 1

    # data_column: list = st.session_state.get('data_column', [])
    st.session_state['data_column'].append(Series(design_params.combined_dict))

    st.success("Design saved!")

def get_saved_designs():
    """Get all saved designs from session state"""
    return st.session_state.get("data_column")
