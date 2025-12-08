import streamlit as st
import pandas as pd
import numpy as np
from anchor_pro.streamlit.core_functions.parameters import ParameterGroup
from anchor_pro.streamlit.core_ui.anchor_geometry_preview import render_anchor_geometry_preview
from anchor_pro.streamlit.core_functions.concrete_anchorage_calcs import evaluate_concrete_anchors
from anchor_pro.streamlit.core_ui.sidebar import load_anchor_catalog

def render_parameters_display():
    """
    Renders a visual list of saved Parameters from the session state,
    calculates results for each, and displays them.
    """
    # Access the group from session state
    group: ParameterGroup = st.session_state.get('parameter_group')

    if not group or not group.parameters:
        st.info("No configurations saved yet.")
        return

    st.markdown("### Saved Configurations")

    # Load Catalog once for all calcs
    df_catalog = load_anchor_catalog()

    # # 1. Render Preview Section
    # render_anchor_geometry_preview(group)
    
    # st.divider()

    # 2. Render Details List with Calc Results
    st.markdown("#### Details & Results")

    for idx, param in enumerate(group.parameters):
        # Perform Calculation
        result = None
        error_msg = None
        
        if param.selected_anchor_id:
            try:
                result = evaluate_concrete_anchors(param, df_catalog)
            except Exception as e:
                error_msg = str(e)
        else:
            error_msg = "No Anchor Selected"

        # Create Header Label with Pass/Fail status
        anchor_lbl = param.selected_anchor_id if param.selected_anchor_id else "No Anchor"
        
        status_icon = "⚪"
        if result:
            status_icon = "✅" if result.ok else "❌"
        elif error_msg:
            status_icon = "⚠️"

        label = f"**{idx + 1}. {param.name}** | {status_icon} | {anchor_lbl}"
        
        # 2. Expander for full details
        with st.expander(label):
            # Show top-level result summary
            if result:
                # Extract governing unities to determine mode
                idx_anc = result.governing_anchor_idx
                idx_th = result.governing_theta_idx
                
                # Retrieve individual component DCRs from the arrays
                dcr_n = result.tension_unity_by_anchor[idx_anc, idx_th]
                
                # Check if tension dominates
                # ACI interaction: (N/Nn)^(5/3) + (V/Vn)^(5/3) = result.unity
                # If result.unity is approximately dcr_n^(5/3), then shear is negligible.
                is_tension = np.isclose(result.unity, dcr_n**(5.0/3.0), rtol=0.01)
                
                mode_str = "Tension" if is_tension else "Shear/Interaction"

                c1, c2, c3 = st.columns(3)
                c1.metric("Unity Check", f"{result.unity:.2f}")
                c2.metric("Status", "PASS" if result.ok else "FAIL")
                c3.markdown(f"**Gov. Mode:** {mode_str}")
                st.divider()
            elif error_msg:
                st.error(f"Calculation Failed: {error_msg}")

            # Organize full details into tabs
            tab_conc, tab_geo, tab_loads = st.tabs(["Concrete & Anchor", "Geometry", "Loads"])

            with tab_conc:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Profile:** {param.profile.value}")
                    st.markdown(f"**Strength (f'c):** {param.fc:.0f} psi")
                    st.markdown(f"**Thickness:** {param.t_slab} in")
                with c2:
                    st.markdown(f"**Weight:** {param.weight_classification}")
                    st.markdown(f"**Cracked:** {param.cracked_concrete}")
                
            with tab_geo:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Fixture:** {param.Bx}\" x {param.By}\"")
                with c2:
                    fmt = lambda x: "∞" if np.isinf(x) else f"{x:.2f}\""
                    st.markdown(f"**Edges:** L:{fmt(param.cx_neg)}, R:{fmt(param.cx_pos)}, B:{fmt(param.cy_neg)}, T:{fmt(param.cy_pos)}")

                if param.xy_anchors is not None:
                    st.markdown("**Anchors:**")
                    df_anch = pd.DataFrame(param.xy_anchors, columns=["x", "y"])
                    st.dataframe(df_anch, hide_index=True, height=100, width='stretch')

            with tab_loads:
                loads = param.loads
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"- **N:** {loads.get('N', 0):.0f} lbs")
                    st.markdown(f"- **Vx:** {loads.get('Vx', 0):.0f} lbs")
                with c2:
                    st.markdown(f"- **Mx:** {loads.get('Mx', 0):.0f} lb-in")
                    st.markdown(f"- **My:** {loads.get('My', 0):.0f} lb-in")