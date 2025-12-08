import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from anchor_pro.streamlit.core_functions.parameters import ParameterGroup
from anchor_pro.streamlit.core_functions.concrete_anchorage_calcs import evaluate_concrete_anchors
from anchor_pro.streamlit.core_ui.sidebar import load_anchor_catalog
from anchor_pro.elements.concrete_anchors import ConcreteAnchorResults, ShearDirLabel

def render_utilization_chart(df: pd.DataFrame):
    """Render horizontal bar chart of utilization ratios"""

    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()

    # Sort by utilization for better visibility
    df_sorted = df.sort_values('Utilization', ascending=True)

    # Color based on utilization level
    colors = ['red' if x > 1.0 else 'orange' if x > 0.8 else 'green'
              for x in df_sorted['Utilization']]

    fig = go.Figure()

    # Add utilization bars
    fig.add_trace(go.Bar(
        y=df_sorted['Limit State'],
        x=df_sorted['Utilization'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.2f}" for x in df_sorted['Utilization']],
        textposition='outside',
        name='Utilization',
        hovertemplate='<b>%{y}</b><br>' +
                      'Utilization: %{x:.2f}<br>' +
                      '<extra></extra>'
    ))

    # Add reference lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                  annotation_text="Capacity Limit", annotation_position="top")
    fig.add_vline(x=0.8, line_dash="dot", line_color="orange",
                  annotation_text="80% Threshold", annotation_position="bottom")

    fig.update_layout(
        title="Limit State Utilization Ratios",
        xaxis_title="Utilization (Demand / Factored Capacity)",
        yaxis_title="",
        height=500,
        xaxis=dict(range=[0, max(1.2, df_sorted['Utilization'].max() * 1.1)]),
        showlegend=False,
        hovermode='y unified'
    )

    st.plotly_chart(fig, width='stretch')

def render_anchor_results_analysis():
    """
    Displays the analysis results for the selected Concrete Anchor Design.
    Retrieves the selected parameter set from session_state['preview_config_select'].
    """
    
    # 1. Retrieve Selected Design and Parameters
    selected_name = st.session_state.get("preview_config_select")
    group = st.session_state.get("parameter_group")
    
    if not selected_name or not group:
        st.info("Please select a design configuration.")
        return

    param = group.get_parameter_by_name(selected_name)
    if not param:
        st.error(f"Design '{selected_name}' not found.")
        return

    if not param.selected_anchor_id:
        st.warning("Please select an anchor product to view analysis results.")
        return

    # 2. Run Evaluation
    df_catalog = load_anchor_catalog()
    try:
        results: ConcreteAnchorResults = evaluate_concrete_anchors(param, df_catalog)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    # 3. Render Tabs
    st.markdown("### Concrete Anchor Analysis Results")
    tab_summary, tab_inputs, tab_details = st.tabs([
        "Utilization Summary", 
        "Input Parameters", 
        "Detailed Results"
    ])

    # --- Tab 1: Utilization Summary ---
    with tab_summary:
        # Collect all utilization data for the chart
        chart_data = []

        # Helper to extract max DCR
        def extract_max_dcr(name, calcs):
            if not calcs:
                return
            
            max_val = -1.0
            found = False
            for c in calcs:
                if c and c.unities is not None and c.unities.size > 0:
                    max_val = max(max_val, np.max(c.unities))
                    found = True
            
            if found and max_val >= 0:
                chart_data.append({"Limit State": name, "Utilization": max_val})

        # 1. Interaction (Overall)
        chart_data.append({"Limit State": "Interaction", "Utilization": results.unity})

        # 2. Tension Limit States
        extract_max_dcr("Steel Tensile Strength", results.steel_tension_calcs)
        extract_max_dcr("Concrete Tension Breakout", results.tension_breakout_calcs)
        extract_max_dcr("Anchor Pullout", results.anchor_pullout_calcs)
        extract_max_dcr("Side Face Blowout", results.side_face_blowout_calcs)
        extract_max_dcr("Bond Strength", results.bond_strength_calcs)

        # 3. Shear Limit States
        extract_max_dcr("Steel Shear Strength", results.steel_shear_calcs)
        extract_max_dcr("Shear Pryout", results.shear_pryout_calcs)
        
        # 4. Shear Breakout by Direction
        for i, calc in enumerate(results.shear_breakout_calcs):
            if i < len(results.shear_groups):
                lbl = results.shear_groups[i].direction.value
                # We assume the calc corresponds to the group at the same index
                # (Logic from existing code structure)
                if calc and calc.unities is not None and calc.unities.size > 0:
                    val = np.max(calc.unities)
                    chart_data.append({"Limit State": f"Shear Breakout ({lbl})", "Utilization": val})

        # Render Chart
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            render_utilization_chart(df_chart)
            
            # Overall Status Indicator below chart
            status_color = "green" if results.ok else "red"
            status_text = "OK" if results.ok else "NG"
            st.markdown(f"**Overall Design Status:** :{status_color}[{status_text}] (Max Utilization: {results.unity:.3f})")
        else:
            st.info("No utilization results calculated.")

    # --- Tab 2: Input Parameters ---
    with tab_inputs:
        st.markdown("#### General Parameters")
        # Build dictionary for inputs
        hef_display = "N/A"
        if results.tension_breakout_calcs:
            hef_display = f"{results.tension_breakout_calcs[0].hef:.2f}"

        input_data = {
            "Design Name": param.name,
            "Anchor ID": param.selected_anchor_id,
            "Profile": param.profile.value,
            "Concrete Strength (f'c)": f"{param.fc:,.0f} psi",
            "Cracked Concrete": str(param.cracked_concrete),
            "Slab Thickness": f"{param.t_slab:.2f} in",
            "Anchor Position": param.anchor_position.value,
            "Number of Anchors": len(param.xy_anchors),
            "Embedment Depth (hef)": hef_display,
            "Load Mode": param.load_mode
        }
        
        # Add Global Loads if applicable
        if param.load_mode == "Global":
            for k, v in param.loads.items():
                if v != 0:
                    input_data[f"Load {k}"] = f"{v:,.0f} lbs" if 'M' not in k else f"{v:,.0f} lb-in"

        df_inputs = pd.DataFrame(input_data.items(), columns=["Parameter", "Value"])
        st.dataframe(df_inputs, hide_index=True)

        st.markdown("#### Anchor Position and Forces")
        st.caption("Detailed geometry and forces for each anchor (Governing Load Case).")

        # Create Table for Anchors and Forces
        # Get Governing Forces based on governing theta
        theta_idx = results.governing_theta_idx
        # forces shape: (n_anchor, 3, n_theta) -> (n_anchor, 3)
        gov_forces = results.forces[:, :, theta_idx]

        # Combine Geometry and Forces
        # param.xy_anchors is (n_anchor, 2)
        anch_data = []
        for i, (xy, f) in enumerate(zip(param.xy_anchors, gov_forces)):
            anch_data.append({
                "Anchor #": i + 1,
                "X (in)": xy[0],
                "Y (in)": xy[1],
                "Tension (lb)": f[0], # N
                "Shear X (lb)": f[1], # Vx
                "Shear Y (lb)": f[2]  # Vy
            })
        
        df_anchors = pd.DataFrame(anch_data)
        st.dataframe(
            df_anchors.style.format({
                "X (in)": "{:.2f}",
                "Y (in)": "{:.2f}",
                "Tension (lb)": "{:,.0f}",
                "Shear X (lb)": "{:,.0f}",
                "Shear Y (lb)": "{:,.0f}"
            }),
            
            hide_index=True
        )

    # --- Tab 3: Detailed Results ---
    with tab_details:
        summary_rows = []
        
        # Reusing similar logic to get max DCR per limit state row
        def add_summary_row(limit_state_name, calcs_list):
            if not calcs_list:
                return
            
            # Find the worst group/anchor for this specific limit state list
            worst_dcr = -1.0
            demand_val = 0.0
            cap_val = 0.0
            
            found_any = False
            for calc in calcs_list:
                if calc is None or calc.unities is None or calc.unities.size == 0:
                    continue
                found_any = True
                curr_max = np.max(calc.unities)
                if curr_max > worst_dcr:
                    worst_dcr = curr_max
                    # Get corresponding demand/cap
                    flat_idx = np.argmax(calc.unities)
                    if calc.demand.ndim > 1:
                        dem = calc.demand.flatten()[flat_idx]
                    else:
                        dem = calc.demand[flat_idx]
                    demand_val = dem
                    cap_val = dem / worst_dcr if worst_dcr > 1e-6 else 0.0
            
            if found_any:
                summary_rows.append({
                    "Limit State": limit_state_name,
                    "Demand (lbs)": demand_val,
                    "Capacity (lbs)": cap_val,
                    "DCR": worst_dcr,
                    "Status": "OK" if worst_dcr <= 1.0 else "NG"
                })

        # Add rows
        add_summary_row("Steel Tensile Strength", results.steel_tension_calcs)
        add_summary_row("Concrete Tension Breakout", results.tension_breakout_calcs)
        add_summary_row("Anchor Pullout", results.anchor_pullout_calcs)
        add_summary_row("Side Face Blowout", results.side_face_blowout_calcs)
        add_summary_row("Bond Strength", results.bond_strength_calcs)
        add_summary_row("Steel Shear Strength", results.steel_shear_calcs)
        add_summary_row("Shear Pryout", results.shear_pryout_calcs)
        
        # Add separate rows for each shear breakout direction
        for i, calc in enumerate(results.shear_breakout_calcs):
            if i < len(results.shear_groups):
                lbl = results.shear_groups[i].direction.value
                add_summary_row(f"Shear Breakout ({lbl})", [calc])

        if summary_rows:
            df_res = pd.DataFrame(summary_rows)
            
            # Formatting
            st.dataframe(
                df_res.style.format({
                    "Demand (lbs)": "{:,.0f}",
                    "Capacity (lbs)": "{:,.0f}",
                    "DCR": "{:.3f}"
                }).map(lambda x: "color: red; font-weight: bold" if x == "NG" else "color: green; font-weight: bold", subset=["Status"]),
                
                hide_index=True
            )
        else:
            st.write("No detailed results available.")