import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Optional, Dict, Any
from core_functions.results_visualization import render_anchor_calculation_results
from core_functions.design_parameters import DesignParameters

def render_visualizations(anchor_data: pd.DataFrame):
    """Render all visualizations for anchor data"""
    if anchor_data is None or anchor_data.empty:
        st.info("No anchor data to visualize. Add anchors in the editor.")
        return
    col1, col2 = st.columns(2)
    with col1:
        render_anchor_layout(anchor_data)
    with col2:
        render_anchor_calculation_results(df = st.session_state["analysis_results_df"])


def render_anchor_layout(df: pd.DataFrame):
    """Render 2D anchor layout visualization with bounding box"""
    fig = go.Figure()
    
    # Add anchor points
    fig.add_trace(go.Scatter(
        x=df['X'],
        y=df['Y'],
        mode='markers+text',
        marker=dict(
            size=15,
            color='blue',
            symbol='circle',
            line=dict(width=2, color='darkblue')
        ),
        text=[f"A{i+1}" for i in range(len(df))],
        textposition="top right",
        name="Anchors",
        hovertemplate='Anchor %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    # Calculate centroid
    centroid_x = df['X'].mean()
    centroid_y = df['Y'].mean()
    
    # Add centroid if multiple anchors
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='x',
                line=dict(width=2, color='darkred')
            ),
            name="Centroid",
            hovertemplate='Centroid<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # Add bounding box (baseplate) from session state
    if 'data_column' in st.session_state and st.session_state['data_column']:
        Bx = st.session_state['data_column'][0].get('Bx', 24.0)  # Default to 24 if not found
        By = st.session_state['data_column'][0].get('By', 24.0)  # Default to 24 if not found
        
        # Calculate box corners centered on centroid
        half_width = Bx / 2
        half_height = By / 2
        
        box_x = [
            centroid_x - half_width,  # Bottom-left
            centroid_x + half_width,  # Bottom-right
            centroid_x + half_width,  # Top-right
            centroid_x - half_width,  # Top-left
            centroid_x - half_width   # Close the box
        ]
        
        box_y = [
            centroid_y - half_height,  # Bottom-left
            centroid_y - half_height,  # Bottom-right
            centroid_y + half_height,  # Top-right
            centroid_y + half_height,  # Top-left
            centroid_y - half_height   # Close the box
        ]
        
        # Add the bounding box
        fig.add_trace(go.Scatter(
            x=box_x,
            y=box_y,
            mode='lines',
            line=dict(
                color='green',
                width=2,
                dash='solid'
            ),
            name=f"Baseplate ({Bx}\" × {By}\")",
            hovertemplate='Baseplate Boundary<br>Width: ' + f'{Bx}' + ' in<br>Height: ' + f'{By}' + ' in<extra></extra>'
        ))
        
        # Add corner markers for clarity
        fig.add_trace(go.Scatter(
            x=box_x[:-1],  # Exclude the closing point
            y=box_y[:-1],
            mode='markers',
            marker=dict(
                size=6,
                color='green',
                symbol='square'
            ),
            name="Baseplate Corners",
            showlegend=False,
            hovertemplate='Corner<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # Calculate appropriate axis range to show everything
    if 'data_column' in st.session_state and st.session_state['data_column']:
        Bx = st.session_state['data_column'][0].get('Bx', 24.0)
        By = st.session_state['data_column'][0].get('By', 24.0)
        
        # Get min/max values for proper scaling
        all_x = list(df['X']) + [centroid_x - Bx/2, centroid_x + Bx/2]
        all_y = list(df['Y']) + [centroid_y - By/2, centroid_y + By/2]
        
        x_range = [min(all_x) - 2, max(all_x) + 2]
        y_range = [min(all_y) - 2, max(all_y) + 2]
        
        # Make the range square (equal aspect ratio)
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        if x_span > y_span:
            diff = (x_span - y_span) / 2
            y_range[0] -= diff
            y_range[1] += diff
        else:
            diff = (y_span - x_span) / 2
            x_range[0] -= diff
            x_range[1] += diff
    else:
        x_range = None
        y_range = None
    
    # Update layout
    fig.update_layout(
        title="Anchor Layout with Baseplate",
        xaxis_title="X (in)",
        yaxis_title="Y (in)",
        height=800,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            range=x_range
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            scaleanchor="x",  # This ensures equal aspect ratio
            scaleratio=1,      # 1:1 ratio
            range=y_range
        )
    )
    
    # Add annotations for dimensions
    if 'data_column' in st.session_state and st.session_state['data_column']:
        Bx = st.session_state['data_column'][0].get('Bx', 24.0)
        By = st.session_state['data_column'][0].get('By', 24.0)
        
        # Add dimension annotations
        fig.add_annotation(
            x=centroid_x,
            y=centroid_y - By/2 - 1,
            text=f"{Bx}\"",
            showarrow=False,
            font=dict(size=12, color="green"),
            yshift=-10
        )
        
        fig.add_annotation(
            x=centroid_x - Bx/2 - 1,
            y=centroid_y,
            text=f"{By}\"",
            showarrow=False,
            font=dict(size=12, color="green"),
            xshift=-10,
            textangle=-90
        )
    
    st.plotly_chart(fig, use_container_width=True)

def render_force_distribution(df: pd.DataFrame):
    """Render force distribution among anchors"""
    if not all(col in df.columns for col in ['Vx', 'Vy', 'N']):
        st.warning("Missing force data for visualization")
        return
    
    # Create subplots for different force components
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for forces
        force_data = pd.DataFrame({
            'Anchor': [f"A{i+1}" for i in range(len(df))],
            'Vx': df['Vx'],
            'Vy': df['Vy'],
            'N': df['N']
        })
        
        fig_bar = px.bar(
            force_data.melt(id_vars=['Anchor'], var_name='Force', value_name='Value'),
            x='Anchor',
            y='Value',
            color='Force',
            title='Force Distribution by Anchor',
            barmode='group'
        )
        
        fig_bar.update_layout(
            yaxis_title="Force (lbs)",
            xaxis_title="Anchor ID"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart for tension distribution
        if df['N'].sum() > 0:
            fig_pie = px.pie(
                values=df['N'],
                names=[f"A{i+1}" for i in range(len(df))],
                title='Tension Force Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No tension forces to display")

def render_analysis_results(df: pd.DataFrame):
    """Render analysis results and calculations"""
    
    # Calculate resultant forces
    total_vx = df['Vx'].sum()
    total_vy = df['Vy'].sum()
    total_n = df['N'].sum()
    resultant_v = np.sqrt(total_vx**2 + total_vy**2)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vx", f"{total_vx:.1f} lbs")
    with col2:
        st.metric("Total Vy", f"{total_vy:.1f} lbs")
    with col3:
        st.metric("Total N", f"{total_n:.1f} lbs")
    with col4:
        st.metric("Resultant V", f"{resultant_v:.1f} lbs")
    
    # Mathematical formulas display
    if st.checkbox("Show Calculations"):
        render_calculation_display(df)
    

def render_calculation_display(df: pd.DataFrame):
    """Display mathematical calculations and formulas"""
    
    st.markdown("### Force Equilibrium Check")
    
    # Example calculations
    st.markdown(r"""
    **Resultant Shear Force:**
    $$V_{\text{resultant}} = \sqrt{V_x^2 + V_y^2}$$
    """)
    
    # Show actual calculation
    total_vx = df['Vx'].sum()
    total_vy = df['Vy'].sum()
    resultant = np.sqrt(total_vx**2 + total_vy**2)
    
    # st.markdown(f"""
    # $$V_{{\\text{{resultant}}}} = \\sqrt{{{total_vx:.1f}}^2 + {total_vy:.1f}^2}} = {resultant:.1f} \\text{{ lbs}}$$
    # """)

    st.markdown(f"""
    $$
    V_{{\\text{{resultant}}}} = \\sqrt{{{total_vx:.1f}^2 + {total_vy:.1f}^2}} = {resultant:.1f}\\ \\text{{lbs}}
    $$
    """)


    # Add more calculations as needed
    if len(df) > 1:
        st.markdown("### Moment Equilibrium")
        centroid_x = df['X'].mean()
        centroid_y = df['Y'].mean()

        moments_x = ((df['Y'] - centroid_y) * df['N']).sum()
        moments_y = ((df['X'] - centroid_x) * df['N']).sum()

        st.markdown(f"""
        **Moments about centroid:**
        - $M_x = {moments_x:.1f}$ lb-in
        - $M_y = {moments_y:.1f}$ lb-in
        """)

def switch_active_design(design_index: int):
    """
    Switch the active design by moving the selected design to index 0.
    
    Parameters:
    -----------
    design_index : int
        The index of the design to make active
    """
    if 'data_column' not in st.session_state or not st.session_state['data_column']:
        st.error("No designs available to switch")
        return
    
    designs = st.session_state['data_column']
    
    # Validate index
    if design_index < 0 or design_index >= len(designs):
        st.error(f"Invalid design index: {design_index}")
        return
    
    # If already active, no need to switch
    if design_index == 0:
        st.info("This design is already active")
        return
    
    # Store the selected design
    selected_design = designs[design_index].copy(deep=True)
    
    # Remove it from its current position
    designs.pop(design_index)
    
    # Insert it at index 0 (making it the active design)
    designs.insert(0, selected_design)
    
    # Update session state
    st.session_state['data_column'] = designs
    
    st.success(f"Design {design_index + 1} is now active")
    st.rerun()

def render_project_designs_table():
    """Render the project designs summary table with design switching functionality"""
    with st.expander('Project Designs Summary', expanded=True):
        designs: list = st.session_state.get("data_column", [])
        
        if not designs:
            st.info("No designs saved yet. Use 'Record Data' to save designs.")
            return
        
        # Create tabs for different views
        
        # Convert to DataFrame - each Series becomes a row
        df = pd.DataFrame(designs)
        
        # Add design labels as row index
        df.index = [f"Design {i+1} {'(Active)' if i == 0 else ''}" for i in range(len(designs))]
        
        # Display the dataframe transposed (designs as columns, parameters as rows)
        st.dataframe(
            data=df.T, 
            height=1000,
            use_container_width=True
        )
        
        # Show key parameters for quick comparison
        if len(designs) > 1:
            st.subheader("Quick Comparison")
            comparison_params = ['fc', 'specified_product', 'Bx', 'By', 't_slab', 'cracked_concrete']
            
            comparison_data = []
            for i, design in enumerate(designs):
                row_data = {param: design.get(param, 'N/A') for param in comparison_params}
                comparison_data.append(row_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.index = [f"Design {i+1} {'(Active)' if i == 0 else ''}" for i in range(len(designs))]
            st.dataframe(comparison_df, use_container_width=True)
        
        st.subheader("Select Design to Activate")
        
        if len(designs) <= 1:
            st.info("You need at least 2 designs to switch between them.")
        else:
            # Create selection interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create options for selectbox
                design_options = []
                for i in range(len(designs)):
                    # Get some key info for display
                    fc = designs[i].get('fc', 'N/A')
                    product = designs[i].get('specified_product', 'N/A')
                    status = " (Currently Active)" if i == 0 else ""
                    design_options.append(f"Design {i+1}: fc={fc}, Product={product}{status}")
                
                selected_design_str = st.selectbox(
                    "Choose design to make active:",
                    options=design_options,
                    index=0,
                    key="design_selector"
                )
                
                # Extract the design index from the selected string
                selected_index = int(selected_design_str.split(":")[0].split()[1]) - 1
            
            with col2:
                if st.button("🔄 Make Active", type="primary", use_container_width=True):
                    switch_active_design(selected_index)
            
            # Display preview of selected design
            if selected_index != 0:
                with st.expander("Preview Selected Design", expanded=True):
                    preview_data = designs[selected_index].to_dict()
                    
                    # Show key parameters
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Substrate:**")
                        st.write(f"- fc: {preview_data.get('fc', 'N/A')} psi")
                        st.write(f"- Thickness: {preview_data.get('t_slab', 'N/A')} in")
                        st.write(f"- Cracked: {preview_data.get('cracked_concrete', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Anchor:**")
                        st.write(f"- Product: {preview_data.get('specified_product', 'N/A')}")
                        st.write(f"- Seismic: {preview_data.get('seismic', 'N/A')}")
                    
                    with col3:
                        st.markdown("**Base Plate:**")
                        st.write(f"- Width: {preview_data.get('Bx', 'N/A')} in")
                        st.write(f"- Length: {preview_data.get('By', 'N/A')} in")
                    
                    # Show anchor geometry if available
                    if 'anchor_geometry_forces' in preview_data:
                        st.markdown("**Anchor Geometry:**")
                        anchor_df = preview_data['anchor_geometry_forces']
                        if isinstance(anchor_df, pd.DataFrame) and not anchor_df.empty:
                            st.dataframe(anchor_df, use_container_width=True)
    
        # Clear all designs button
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear All Designs", type="secondary", use_container_width=True):
                if st.session_state.get("clear_confirm", False):
                    st.session_state['data_column'] = clear_designs()
                    st.session_state['data_column_counter'] = 0
                    st.session_state['clear_confirm'] = False
                    st.success("All designs cleared!")
                    st.rerun()
                else:
                    st.session_state['clear_confirm'] = True
                    st.warning("Click again to confirm clearing all designs")
        
        with col2:
            if st.button("❌ Remove Active", use_container_width=True):
                if len(designs) > 1:
                    designs.pop(0)
                    st.session_state['data_column'] = designs
                    st.success("Active design removed")
                    st.rerun()
                else:
                    st.warning("Cannot remove the only design")
        
        # Reset confirmation if user clicks elsewhere
        if st.session_state.get("clear_confirm", False):
            if not st.button("Cancel", key="cancel_clear"):
                st.session_state['clear_confirm'] = False

def clear_designs():
    """Clears designs and resets to default data column"""
    default_data_column: list = []
    design_params = DesignParameters()

    default_series = pd.Series(design_params.combined_dict)
    default_data_column.insert(0, default_series)
    return default_data_column