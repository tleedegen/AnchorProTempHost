import streamlit as st
from pandas import Series
from models.anchor_design_data import DesignParameters

def app_setup(design_params: DesignParameters):
    """Initialize session state variables"""
    if "data_column" not in st.session_state:
        save_design_to_session(design_params)

def save_design_to_session(design_params: DesignParameters):
    """Save current design to session state"""
    if "data_column_counter" not in st.session_state:
        st.session_state["data_column_counter"] = 0
    st.session_state["data_column_counter"] = st.session_state["data_column_counter"] + 1

    # data_column: list = st.session_state.get('data_column', [])
    if "data_column" not in st.session_state:
        st.session_state["data_column"] = []
    st.session_state['data_column'].append(Series(design_params.combined_dict))
    if st.session_state['data_column_counter'] >= 2:
        st.success("Design saved!")

def get_saved_designs():
    """Get all saved designs from session state"""
    return st.session_state.get("data_column")

def update_active_anchor_geometry_session_state(data_column_key: str, data: float):
    """Update session state anchor geometry. Active parameters is the first index of the session_state[data_column]"""
    if 'data_column' in st.session_state:
        st.session_state['data_column'][0][data_column_key] = data
