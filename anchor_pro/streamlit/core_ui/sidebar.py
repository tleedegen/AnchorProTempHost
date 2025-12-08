import streamlit as st
import pandas as pd
import numpy as np
import os
from anchor_pro.streamlit.core_functions.user_inputs import (
    render_concrete_properties, 
    render_anchor_geometry_and_loads, 
    render_anchor_selector
)
from anchor_pro.streamlit.auth.login_ui import render_login_sidebar
from anchor_pro.config import base_path
from anchor_pro.streamlit.core_functions.parameters import (
    Parameters, 
    ParameterGroup, 
    initialize_parameter_group
)
from anchor_pro.elements.concrete_anchors import Profiles

@st.cache_data
def load_anchor_catalog():
    # Adjust path as necessary relative to your entry point
    path = os.path.join(base_path, 'data', 'processed', 'anchors.parquet')
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame() 

def load_parameter_to_session_state(param: Parameters):
    """
    Maps a saved Parameters object back to the specific st.session_state keys
    used by the input widgets in user_inputs.py.
    """
    # 1. Concrete Properties
    st.session_state['concrete_profile'] = param.profile.value
    st.session_state['weight_classification'] = param.weight_classification
    st.session_state['cracked_concrete'] = param.cracked_concrete
    st.session_state['concrete_strength'] = param.fc
    st.session_state['lw_factor'] = param.lw_factor
    st.session_state['slab_thickness'] = param.t_slab
    st.session_state['poisson_ratio'] = param.poisson

    # 2. Anchor Selection
    # The selector widget uses the ID. 
    # Note: If the ID is not in the catalog currently loaded, selectbox might warn or error 
    # unless handled, but usually streamit defaults gracefully if index not found.
    st.session_state['selected_anchor_id'] = param.selected_anchor_id

    # 3. Geometry & Loads
    st.session_state['load_distribution_mode'] = "Rigid Plate (Global Loads)" if param.load_mode == "Global" else "Individual Anchors (Direct Assignment)"
    
    # Global Loads
    # Default 0.0 if not present
    st.session_state['load_N'] = param.loads.get("N", 0.0)
    st.session_state['load_Vx'] = param.loads.get("Vx", 0.0)
    st.session_state['load_Vy'] = param.loads.get("Vy", 0.0)
    st.session_state['load_Mx'] = param.loads.get("Mx", 0.0)
    st.session_state['load_My'] = param.loads.get("My", 0.0)
    st.session_state['load_T'] = param.loads.get("T", 0.0)

    # Fixture Dimensions
    st.session_state['geo_Bx'] = param.Bx
    st.session_state['geo_By'] = param.By
    st.session_state['geo_anchor_pos'] = param.anchor_position.value

    # Edge Distances
    # Logic in render_edge_input uses a checkbox (inf_...) and a value input (val_...)
    def set_edge_state(name, val):
        if np.isinf(val):
            st.session_state[f'inf_{name}'] = True
            # We don't set val_... strictly needed, but keeps UI clean
        else:
            st.session_state[f'inf_{name}'] = False
            st.session_state[f'val_{name}'] = val

    set_edge_state('cx_neg', param.cx_neg)
    set_edge_state('cx_pos', param.cx_pos)
    set_edge_state('cy_neg', param.cy_neg)
    set_edge_state('cy_pos', param.cy_pos)

    # Anchor Data Table (DataFrame reconstruction)
    # xy_anchors is (n, 2). individual_forces is (n, 3) or None.
    n_anchors = param.xy_anchors.shape[0]
    
    data = {
        "x": param.xy_anchors[:, 0],
        "y": param.xy_anchors[:, 1]
    }
    
    if param.individual_forces is not None and len(param.individual_forces) == n_anchors:
        # individual_forces structure assumed [N, Vx, Vy] based on extraction logic
        data["N"] = param.individual_forces[:, 0]
        data["Vx"] = param.individual_forces[:, 1]
        data["Vy"] = param.individual_forces[:, 2]
    else:
        # Defaults if global mode or missing
        data["N"] = [0.0] * n_anchors
        data["Vx"] = [0.0] * n_anchors
        data["Vy"] = [0.0] * n_anchors

    st.session_state['anchor_table_df'] = pd.DataFrame(data)

def render_sidebar():
    # Ensure our data structure exists
    initialize_parameter_group()
    group: ParameterGroup = st.session_state['parameter_group']

    with st.sidebar:
        render_login_sidebar()

        # --- Manage Configurations (Load) ---
        if group.parameters:
            with st.expander("📂 Manage Configurations", expanded=False):
                names = group.get_names()
                selected_load = st.selectbox("Select to Load", options=names, key="load_selector")
                
                if st.button("Load Configuration", width='stretch'):
                    param_to_load = group.get_parameter_by_name(selected_load)
                    if param_to_load:
                        load_parameter_to_session_state(param_to_load)
                        # Set the save name to the loaded name so Overwrite works immediately
                        # st.text_input uses key 'save_config_name' logic below? 
                        # We cannot set widget value easily if key is used, unless we use key in state.
                        # Let's verify how save name is rendered below.
                        st.session_state['config_name_input'] = param_to_load.name
                        st.rerun()

        st.header("Design Inputs")
        
        # 1. Properties (Writes directly to session_state keys)
        render_concrete_properties()
        
        # 2. Geometry AND Loads (Unified)
        geo_load_inputs = render_anchor_geometry_and_loads()
        
        # 3. Anchor Selection (Returns Series)
        df_catalog = load_anchor_catalog()
        anchor_data = render_anchor_selector(df_catalog)
        
        # Store essential data in session state for the main page to access
        st.session_state['geo_inputs'] = geo_load_inputs
        st.session_state['anchor_data_selected'] = anchor_data
        
        # Backward compatibility
        st.session_state['load_inputs'] = geo_load_inputs.get('global_loads', {})

        st.markdown("---")
        st.subheader("Save Configuration")
        
        # Input for the save name
        # Use a key so we can populate it when loading
        if 'config_name_input' not in st.session_state:
            st.session_state['config_name_input'] = "Design 1"
            
        save_name = st.text_input("Configuration Name", key="config_name_input")
        
        # Check if name exists
        name_exists = save_name in group.get_names()

        col1, col2 = st.columns(2)
        
        # Prepare the Parameter object (shared logic)
        def create_current_parameter(name):
             # 1. Gather Concrete Properties from Session State 
            prof_val = st.session_state.get('concrete_profile')
            profile_enum = next(p for p in Profiles if p.value == prof_val)
            
            weight_class = st.session_state.get('weight_classification')
            cracked = st.session_state.get('cracked_concrete')
            fc_val = st.session_state.get('concrete_strength')
            lw_fact = st.session_state.get('lw_factor')
            t_val = st.session_state.get('slab_thickness')
            poisson_val = st.session_state.get('poisson_ratio')

            # 2. Gather Anchor ID
            sel_anchor_id = anchor_data['anchor_id'] if anchor_data is not None else None

            # 3. Create Object
            return Parameters(
                name=name,
                profile=profile_enum,
                weight_classification=weight_class,
                cracked_concrete=cracked,
                fc=fc_val,
                lw_factor=lw_fact,
                t_slab=t_val,
                poisson=poisson_val,
                
                # Unpack Geometry
                xy_anchors=geo_load_inputs['xy_anchors'],
                Bx=geo_load_inputs['Bx'],
                By=geo_load_inputs['By'],
                cx_neg=geo_load_inputs['cx_neg'],
                cx_pos=geo_load_inputs['cx_pos'],
                cy_neg=geo_load_inputs['cy_neg'],
                cy_pos=geo_load_inputs['cy_pos'],
                anchor_position=geo_load_inputs['anchor_position'],
                
                selected_anchor_id=sel_anchor_id,
                
                # Unpack Load Data
                load_mode=geo_load_inputs['load_mode'],
                loads=geo_load_inputs['global_loads'],
                individual_forces=geo_load_inputs['individual_forces']
            )

        with col1:
            # SAVE NEW Button
            # Disable if name exists (force overwrite) or handle duplicates? 
            # Usually better to have specific actions.
            if st.button("Save New", type="primary", disabled=False):
                if not save_name:
                    st.error("Enter a name.")
                elif name_exists:
                    st.error(f"'{save_name}' already exists. Use Overwrite.")
                else:
                    try:
                        new_param = create_current_parameter(save_name)
                        group.add_parameter(new_param)
                        st.success(f"Saved '{save_name}'!")
                        st.rerun() # Rerun to update lists
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            # OVERWRITE Button
            # Only enable if name exists
            if st.button("Overwrite", type="secondary", disabled=not name_exists):
                if not save_name:
                    st.error("Enter a name.")
                else:
                    try:
                        new_param = create_current_parameter(save_name)
                        updated = group.update_parameter(new_param)
                        if updated:
                            st.success(f"Updated '{save_name}'!")
                            st.rerun()
                        else:
                            st.error("Could not find configuration to update.")
                    except Exception as e:
                        st.error(f"Error: {e}")