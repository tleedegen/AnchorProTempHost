import streamlit as st
import pandas as pd
import numpy as np
from models.anchor_design_data import SubstrateParams
from utils.session_state import update_active_data_column



def render_data_editor():
    """Render the anchor geometry and loads editor"""
    
    # Initialize with empty dataframe if not exists
    if "anchor_data" not in st.session_state:
        st.session_state.anchor_data = pd.DataFrame({
            'X': [0.0],
            'Y': [0.0],
            'Vx': [0.0],
            'Vy': [0.0],
            'N': [0.0]
        })

    substrate_param = SubstrateParams()
    bx = st.number_input(
        label = substrate_param.SUBSTRATE_FIELDS["Bx"]["label"],
        min_value = substrate_param.SUBSTRATE_FIELDS["Bx"]["min_value"],
        value = substrate_param.SUBSTRATE_FIELDS["Bx"]["value"],
        key = substrate_param.SUBSTRATE_FIELDS["Bx"]["key"],
    )
    update_active_data_column('Bx', bx)

    by = st.number_input(
        label = substrate_param.SUBSTRATE_FIELDS["By"]["label"],
        min_value = substrate_param.SUBSTRATE_FIELDS["By"]["min_value"],
        value = substrate_param.SUBSTRATE_FIELDS["By"]["value"],
        key = substrate_param.SUBSTRATE_FIELDS["By"]["key"],
    )
    update_active_data_column('By', by)
    st.subheader("Anchor Geometry & Forces")
    
    # Create column configuration for better data entry
    column_config = {
        "X": st.column_config.NumberColumn(
            "X (in)",
            help="X-coordinate of anchor",
            min_value=None,
            max_value=None,
        ),
        "Y": st.column_config.NumberColumn(
            "Y (in)", 
            help="Y-coordinate of anchor",
            min_value=None,
            max_value=None,
        ),
        "Vx": st.column_config.NumberColumn(
            "Vx (lbs)",
            help="Shear force in X direction",
            min_value=None,
            max_value=None
        ),
        "Vy": st.column_config.NumberColumn(
            "Vy (lbs)",
            help="Shear force in Y direction",
            min_value=None,
            max_value=None
        ),
        "N": st.column_config.NumberColumn(
            "N (lbs)",
            help="Tension force (positive = tension)",
            min_value=None,
            max_value=None,
            step=1.0,
        )
    }
    # Create the data editor - use the return value directly
    anchor_geometry_df: pd.DataFrame = st.data_editor(
        st.session_state.anchor_data,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        key="anchor_editor",  # Different key to avoid conflicts
    )
    # Update session state with the edited dataframe
    update_active_data_column('anchor_geometry_df', anchor_geometry_df)
    # st.session_state['data_column'][0]['anchor_geometry_df'] = anchor_geometry_df

    #TODO: Get Centroid from concrete_anchors
    # Display summary information
    if len(anchor_geometry_df) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Anchors", len(anchor_geometry_df))
        with col2:
            if len(anchor_geometry_df) > 1:
                cx = anchor_geometry_df['X'].mean()
                st.metric("Centroid X", f"{cx:.2f} in")
        with col3:
            if len(anchor_geometry_df) > 1:
                cy = anchor_geometry_df['Y'].mean()
                st.metric("Centroid Y", f"{cy:.2f} in")
    
    # Add validation warnings
    if len(anchor_geometry_df) > 1:
        # Check for minimum spacing
        min_spacing = check_minimum_spacing(anchor_geometry_df)
        if min_spacing < 3.0:  # Typical minimum spacing requirement
            st.warning(f"⚠️ Minimum anchor spacing is {min_spacing:.2f} inches. Consider minimum 3.0 inches.")
    
    # Quick action buttons - Streamlit will automatically rerun when session state changes
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add Standard 4-Anchor Pattern", key="btn_add_4_anchor"):
            st.session_state.anchor_data = pd.DataFrame({
                'X': [0.0, 6.0, 6.0, 0.0],
                'Y': [0.0, 0.0, 6.0, 6.0],
                'Vx': [0.0, 0.0, 0.0, 0.0],
                'Vy': [0.0, 0.0, 0.0, 0.0],
                'N': [0.0, 0.0, 0.0, 0.0]
            })
    
    with col2:
        if st.button("Clear All Anchors", key="btn_clear_anchors"):
            st.session_state.anchor_data = pd.DataFrame({
                'X': [0.0],
                'Y': [0.0], 
                'Vx': [0.0],
                'Vy': [0.0],
                'N': [0.0]
            })
    
    with col3:
        if st.button("Center at Origin", key="btn_center_anchors"):
            if len(anchor_geometry_df) > 0:
                centered_df = anchor_geometry_df.copy()
                cx = centered_df['X'].mean()
                cy = centered_df['Y'].mean()
                centered_df['X'] = centered_df['X'] - cx
                centered_df['Y'] = centered_df['Y'] - cy
                st.session_state.anchor_data = centered_df
    
    return anchor_geometry_df

def check_minimum_spacing(df):
    """Check minimum spacing between anchors"""
    if len(df) < 2:
        return float('inf')
    
    min_spacing = float('inf')
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dx = df.iloc[i]['X'] - df.iloc[j]['X']
            dy = df.iloc[i]['Y'] - df.iloc[j]['Y']
            distance = (dx**2 + dy**2)**0.5
            min_spacing = min(min_spacing, distance)
    
    return min_spacing
