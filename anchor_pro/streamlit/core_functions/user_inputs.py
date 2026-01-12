import streamlit as st
import numpy as np
import pandas as pd
from anchor_pro.elements.concrete_anchors import Profiles, AnchorPosition

# ... [Keep render_concrete_properties and render_anchor_selector as they are] ...

def render_concrete_properties():
    """
    Renders Streamlit widgets for selecting concrete properties.
    Returns a dictionary matching the fields of the ConcreteProps dataclass.
    """
    st.subheader("Concrete Properties")

    # Profile Selection mapped to Profiles Enum
    profile_label = st.selectbox(
        "Concrete Profile",
        options=[p.value for p in Profiles],
        index=0,  # Default to Slab
        help="Select the concrete member profile (e.g., Slab, Wall, Filled Deck).",
        key='concrete_profile'
    )

    # Weight Classification
    weight_classification = st.selectbox(
        "Weight Classification",
        options=["NWC", "LWC"],
        index=0, # Default to Normal Weight Concrete
        help="NWC: Normal Weight Concrete, LWC: Lightweight Concrete",
        key='weight_classification'
    )

    # Cracked Concrete
    cracked_concrete = st.checkbox(
        "Cracked Concrete",
        value=True,
        help="Assume concrete is cracked for calculation purposes (conservative).",
        key='cracked_concrete'
    )
    # Concrete Strength (fc)
    fc = st.number_input(
        "Compressive Strength (f'c) [psi]",
        min_value=2000.0,
        max_value=12000.0,
        value=4000.0,
        step=500.0,
        format="%.0f",
        key='concrete_strength',
    )

    # Lightweight Factor (lambda)
    if 'lw_factor' not in st.session_state:
        st.session_state['lw_factor'] = 1.0
    st.session_state['lw_factor'] = 1.0 if weight_classification == "NWC" else 0.75

    # Slab Thickness
    t_slab = st.number_input(
        "Member Thickness (h) [in]",
        min_value=0.0,
        value=6.0,
        step=0.5,
        help="Thickness of the concrete member.",
        key='slab_thickness'
    )

    # Poisson's Ratio (Advanced setting, usually hidden or defaulted)
    poisson = st.number_input(
        "Poisson's Ratio",
        min_value=0.1,
        max_value=0.3,
        value=0.2,
        step=0.05,
        key='poisson_ratio'
    )

def render_anchor_selector(df_catalog: pd.DataFrame):
    """
    Renders a selectbox to pick an anchor from the provided DataFrame.
    Returns the row (pd.Series) corresponding to the selected anchor.
    """
    st.subheader("Anchor Selection")
    
    if df_catalog is None or df_catalog.empty:
        st.error("Anchor catalog not found.")
        return None
    
    anchor_ids = df_catalog['anchor_id'].unique().tolist()
    
    selected_id = st.selectbox(
        "Select Anchor Product",
        options=anchor_ids,
        key="selected_anchor_id"
    )
    
    # Return the row data for the selected anchor
    if selected_id:
        return df_catalog[df_catalog['anchor_id'] == selected_id].iloc[0]
    return None

def render_anchor_geometry_and_loads():
    """
    Combined function to render Anchor Geometry and Load inputs.
    Allows toggling between Global Loads (Rigid Plate) and Individual Anchor Loads.
    """
    with st.expander("Anchor Geometry & Loads", expanded=True):

        st.subheader("Anchor Geometry & Loads")

        # --- Load Distribution Mode ---
        load_mode = st.radio(
            "Load Distribution Method",
            options=["Rigid Plate (Global Loads)", "Individual Anchors (Direct Assignment)"],
            index=0,
            key="load_distribution_mode"
        )
        is_global_mode = load_mode == "Rigid Plate (Global Loads)"

        # --- 1. Global Loads (Only visible in Rigid Plate Mode) ---
        global_loads = {}
        if is_global_mode:
            st.markdown("##### Global Factored Loads")
            st.caption("Loads applied at the center of the fixture.")
            col1, col2 = st.columns(2)
            with col1:
                global_loads["N"] = st.number_input("Tension (+z) [lbs]", value=1000.0, step=100.0, key="load_N")
                global_loads["Vx"] = st.number_input("Shear X (+x) [lbs]", value=500.0, step=100.0, key="load_Vx")
                global_loads["Mx"] = st.number_input("Moment X (+mx) [lb-in]", value=0.0, step=1000.0, key="load_Mx")
            with col2:
                global_loads["T"] = st.number_input("Torsion (+mz) [lb-in]", value=0.0, step=1000.0, key="load_T")
                global_loads["Vy"] = st.number_input("Shear Y (+y) [lbs]", value=0.0, step=100.0, key="load_Vy")
                global_loads["My"] = st.number_input("Moment Y (+my) [lb-in]", value=0.0, step=1000.0, key="load_My")
        else:
            # Zero out global loads if in direct mode
            global_loads = {k: 0.0 for k in ["N", "Vx", "Vy", "Mx", "My", "T"]}

        # --- 2. Anchor Table (Geometry + Loads if Direct) ---
        st.markdown("##### Anchor Geometry & Forces")
        
        # Initialize session state for anchors table if needed
        if 'anchor_table_df' not in st.session_state:
            # Default 4-bolt pattern
            st.session_state['anchor_table_df'] = pd.DataFrame({
                "x": [-3.0, 3.0, -3.0, 3.0],
                "y": [3.0, 3.0, -3.0, -3.0],
                "Vx": [0.0, 0.0, 0.0, 0.0],
                "Vy": [0.0, 0.0, 0.0, 0.0],
                "N": [250.0, 250.0, 250.0, 250.0] # Default forces
            })

        # Prepare Column Configuration based on mode
        column_config = {
            "x": st.column_config.NumberColumn("X (in)", format="%.2f"),
            "y": st.column_config.NumberColumn("Y (in)", format="%.2f"),
        }
        
        # Determine which columns to show/hide/disable
        df_to_show = st.session_state['anchor_table_df'].copy()
        
        if is_global_mode:
            # In global mode, hide force columns or make them read-only/empty
            # For simplicity, we just show Geometry to avoid confusion
            df_to_show = df_to_show[['x', 'y']] 
        else:
            # In individual mode, show and format force columns
            column_config.update({
                "Vx": st.column_config.NumberColumn("Vx (lbs)", format="%.0f"),
                "Vy": st.column_config.NumberColumn("Vy (lbs)", format="%.0f"),
                "N": st.column_config.NumberColumn("N (lbs)", format="%.0f"),
            })

        # Render Data Editor
        edited_df = st.data_editor(
            df_to_show,
            num_rows="dynamic",
            key='anchor_editor_widget',
            column_config=column_config
        )
        
        # Sync back to session state (Merging updates)
        if is_global_mode:
            # If we only edited geometry, preserve the force columns in state (though they might be stale)
            # or reset them. For safety, we update x/y and keep structure.
            st.session_state['anchor_table_df'] = edited_df.join(
                st.session_state['anchor_table_df'][['Vx', 'Vy', 'N']], how='left'
            ).fillna(0.0)
        else:
            st.session_state['anchor_table_df'] = edited_df

        # Extract Data for Returns
        full_df = st.session_state['anchor_table_df']
        xy_anchors = full_df[['x', 'y']].to_numpy()
        
        # If individual mode, extract forces array (N, Vx, Vy)
        if not is_global_mode:
            # Create (n, 3) array [N, Vx, Vy]
            # Note: The dataframe has Vx, Vy, N. We usually pass [N, Vx, Vy] or similar.
            # Let's standardize on [N, Vx, Vy] for internal processing
            individual_forces = full_df[['N', 'Vx', 'Vy']].to_numpy()
        else:
            individual_forces = None

        # --- 3. Fixture Dimensions ---
        st.header("Fixture Dimensions")
        col_dims = st.columns(2)
        with col_dims[0]:
            Bx = st.number_input("Fixture Width (Bx) [in]", min_value=0.0, value=10.0, step=0.5, key="geo_Bx")
        with col_dims[1]:
            By = st.number_input("Fixture Height (By) [in]", min_value=0.0, value=10.0, step=0.5, key="geo_By")

        # --- 4. Edge Distances ---
        st.header("Concrete Edge Distances")
        # Helper function (Same as before)
        def render_edge_input(label, key_suffix):
            col_check, col_val = st.columns([0.4, 0.6])
            with col_check:
                is_inf = st.checkbox(f"{label}", value=True, key=f"inf_{key_suffix}", help=f"Unbounded {label}", label_visibility="visible")
            with col_val:
                if is_inf:
                    st.text_input(f"Dist. {label}", value="∞", disabled=True, label_visibility="collapsed", key=f"disp_{key_suffix}")
                    return np.inf
                else:
                    return st.number_input(f"Distance {label}", min_value=0.0, value=12.0, step=1.0, label_visibility="collapsed", key=f"val_{key_suffix}")
        st.subheader("Check box for unbounded (∞) edges")
        c1, c2 = st.columns(2)
        with c1:
            cx_neg = render_edge_input("(-X)", "cx_neg")
            cy_neg = render_edge_input("(-Y)", "cy_neg")
        with c2:
            cx_pos = render_edge_input("(+X)", "cx_pos")
            cy_pos = render_edge_input("(+Y)", "cy_pos")

        # --- 5. Anchor Position ---
        st.header("Installation Position")
        pos_label = st.selectbox(
            "Anchor Position",
            options=[p.value for p in AnchorPosition],
            index=0,
            key="geo_anchor_pos"
        )
        anchor_position = next(p for p in AnchorPosition if p.value == pos_label)

        return {
            "load_mode": "Global" if is_global_mode else "Individual",
            "global_loads": global_loads,
            "individual_forces": individual_forces,
            "xy_anchors": xy_anchors,
            "Bx": Bx, "By": By,
            "cx_neg": cx_neg, "cx_pos": cx_pos,
            "cy_neg": cy_neg, "cy_pos": cy_pos,
            "anchor_position": anchor_position
        }