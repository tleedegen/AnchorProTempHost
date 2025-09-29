import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import streamlit as st
import numpy as np
import pandas as pd
from concrete_anchors import ConcreteAnchors

@st.cache_data
def load_anchor_spec_sheet() -> pd.DataFrame:
    """Load anchor data from parquet file with caching"""
    try:
        file_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "anchors.parquet"
        df = pd.read_parquet(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Anchor data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading anchor data: {str(e)}")
        return pd.DataFrame()

# @st.cache_data
def get_manufacturers(df: pd.DataFrame) -> Tuple[str, ...]:
    """Get unique manufacturers from anchor data"""
    if df.empty:
        return tuple()
    return tuple(df["manufacturer"].dropna().unique())

@st.cache_data
def get_anchor_products(df: pd.DataFrame, manufacturer: Optional[str] = None) -> Tuple[str, ...]:
    """Get anchor products, optionally filtered by manufacturer"""
    if df.empty:
        return tuple()

    if manufacturer:
        filtered_df = df[df["manufacturer"] == manufacturer]
        return tuple(filtered_df["anchor_id"].dropna().unique())
    else:
        return tuple(df["anchor_id"].dropna().unique())


def anchor_pro_anchors(df: pd.DataFrame) -> list[list[float]]:
    """Select required anchors coordinates for anchor pro"""
    anchor_coords = df[['X', 'Y']].to_numpy().tolist()
    return anchor_coords

def anchor_pro_forces(df: pd.DataFrame):
    """Select required columns and convert to nested list"""
    anchor_pro_force = df[['N', 'Vx', 'Vy']].to_numpy()[:, np.newaxis, :]
    return anchor_pro_force

def anchor_pro_concrete_data(session_state_data_column: pd.Series) -> pd.Series:
    """All AnchorPro parameters sent to the backend are listed here"""
    required_data: tuple = (
        'Bx', 'By', 'fc', 'lw_factor', 'cracked_concrete', 'poisson', 't_slab',
        'cx_neg', 'cx_pos', 'cy_neg', 'cy_pos', 'profile', 'anchor_position',
        'weight_classification_base'
    )
    anchor_pro_series = pd.Series({k: session_state_data_column[k] for k in required_data})
    return anchor_pro_series

def anchor_pro_set_data():
    """Set all data for AnchorPro calculations and store results in session state"""

    if len(st.session_state['data_column']) > 1:
        # Ensure active index is valid
        if st.session_state['active_data_column_index'] >= len(st.session_state['data_column']):
            st.session_state['active_data_column_index'] = 1

        concrete_data = anchor_pro_concrete_data(st.session_state['data_column'][st.session_state['active_data_column_index']])
        xy_anchors = anchor_pro_anchors(st.session_state['data_column'][st.session_state['active_data_column_index']]["anchor_geometry_forces"])
        anchor_specs = load_anchor_spec_sheet()
        anchor_id = st.session_state['data_column'][st.session_state['active_data_column_index']]["specified_product"]
        anchor_data = anchor_specs[anchor_specs['anchor_id']==anchor_id].iloc[0]

        model = ConcreteAnchors()
        model.set_data(concrete_data, xy_anchors)
        model.set_mechanical_anchor_properties(anchor_data)
        model.anchor_forces = anchor_pro_forces(st.session_state['data_column'][st.session_state['active_data_column_index']]['anchor_geometry_forces'])
        model.check_anchor_spacing()
        model.get_governing_anchor_group()
        model.check_anchor_capacities()

        st.session_state['analysis_results_df'] = model.results

# Export functions for easy import
__all__ = [
    'load_anchor_spec_sheet',
    'get_manufacturers',
    'get_anchor_products',
]