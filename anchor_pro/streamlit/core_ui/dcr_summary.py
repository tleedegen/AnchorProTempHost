import streamlit as st
import pandas as pd
import numpy as np
from anchor_pro.streamlit.core_functions.parameters import ParameterGroup
from anchor_pro.streamlit.core_functions.concrete_anchorage_calcs import evaluate_concrete_anchors
from anchor_pro.streamlit.core_ui.sidebar import load_anchor_catalog
from anchor_pro.elements.concrete_anchors import ConcreteAnchorResults, ShearDirLabel

def _get_max_unity(calc_obj) -> float:
    """Safely extracts the maximum unity value from a Calculation object."""
    if calc_obj is None or calc_obj.unities is None:
        return np.nan
    return np.max(calc_obj.unities)

def _extract_shear_breakout_by_direction(result: ConcreteAnchorResults) -> dict:
    """
    Iterates through shear groups to find the max DCR for each direction.
    Returns a dict like {'Shear Breakout (X+)': 0.5, ...}
    """
    directions = {
        ShearDirLabel.XP: "Shear Breakout (X+)",
        ShearDirLabel.XN: "Shear Breakout (X-)",
        ShearDirLabel.YP: "Shear Breakout (Y+)",
        ShearDirLabel.YN: "Shear Breakout (Y-)",
    }
    
    # Initialize with 0.0 or NaN
    breakout_dcr = {label: 0.0 for label in directions.values()}
    
    if result.shear_groups:
        for idx, group in enumerate(result.shear_groups):
            # Check if this group has a calculated result
            if idx < len(result.shear_breakout_calcs):
                calc = result.shear_breakout_calcs[idx]
                val = _get_max_unity(calc)
                
                # Update the max for this direction
                label = directions.get(group.direction)
                if label and not np.isnan(val):
                    breakout_dcr[label] = max(breakout_dcr[label], val)
                    
    return breakout_dcr

def render_dcr_summary():
    """
    Renders a DCR Summary table for all parameters in the session state.
    Calculates results on the fly to ensure data is current.
    """
    st.markdown("### 📋 Demand/Capacity Ratios Summary")

    group: ParameterGroup = st.session_state.get('parameter_group')
    if not group or not group.parameters:
        st.info("No configurations available. Add a design to see the summary.")
        return

    # Load catalog once
    df_catalog = load_anchor_catalog()

    # Define the rows we want in the table
    limit_states_order = [
        "Steel Tensile Strength",
        "Concrete Tension Breakout",
        "Anchor Pullout",
        "Side Face Blowout",
        "Bond Strength",
        "Steel Shear Strength",
        "Shear Breakout (X+)",
        "Shear Breakout (X-)",
        "Shear Breakout (Y+)",
        "Shear Breakout (Y-)",
        "Shear Pryout",
        "Interaction"
    ]

    # Initialize data dictionary: { "Limit State": [...], "Design 1": [...], ... }
    data = {"Limit State": limit_states_order}

    # Iterate through all designs
    for param in group.parameters:
        col_name = param.name
        column_values = {ls: np.nan for ls in limit_states_order} # Default to NaN

        if param.selected_anchor_id:
            try:
                # Run Calculation
                res = evaluate_concrete_anchors(param, df_catalog)
                tg_idx = res.governing_tension_group

                # Populate Tension Values
                column_values["Steel Tensile Strength"] = _get_max_unity(res.steel_tension_calcs[tg_idx])
                column_values["Concrete Tension Breakout"] = _get_max_unity(res.tension_breakout_calcs[tg_idx])
                column_values["Anchor Pullout"] = _get_max_unity(res.anchor_pullout_calcs[tg_idx])
                column_values["Side Face Blowout"] = _get_max_unity(res.side_face_blowout_calcs[tg_idx])
                column_values["Bond Strength"] = _get_max_unity(res.bond_strength_calcs[tg_idx])

                # Populate Shear Values
                column_values["Steel Shear Strength"] = _get_max_unity(res.steel_shear_calcs[tg_idx])
                column_values["Shear Pryout"] = _get_max_unity(res.shear_pryout_calcs[tg_idx])
                
                # Detailed Shear Breakout by Direction
                breakouts = _extract_shear_breakout_by_direction(res)
                for k, v in breakouts.items():
                    column_values[k] = v

                # Interaction
                column_values["Interaction"] = res.unity

            except Exception:
                # If calculation fails, leave as NaN or mark error (optional)
                pass
        
        # Add column to data dict
        data[col_name] = [column_values[ls] for ls in limit_states_order]

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index("Limit State", inplace=True)

    # --- Styling ---
    def style_dcr(val):
        """
        Colors values > 1.0 Red, <= 1.0 Green.
        NaNs are unstyled (or transparent).
        """
        if pd.isna(val):
            return ""
        if val > 1.0:
            return "background-color: #8B0000; color: white;" # Dark Red
        else:
            return "background-color: #228B22; color: white;" # Forest Green

    # Apply styling
    styler = df.style.map(style_dcr)
    
    # Format numbers to 3 decimal places
    styler = styler.format("{:.3f}")

    # Display
    st.dataframe(styler, width='stretch')