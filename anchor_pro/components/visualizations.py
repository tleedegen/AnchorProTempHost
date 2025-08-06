import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Optional, Dict, Any
from components.ap_results_visualization import render_anchor_calculation_results
from config.constants import PROJECT_DESIGN_COLUMNS

def render_visualizations(anchor_data: pd.DataFrame):
    """Render all visualizations for anchor data"""
    if anchor_data is None or anchor_data.empty:
        st.info("No anchor data to visualize. Add anchors in the editor.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Anchor Layout", "Force Distribution", "Analysis Results"])
    
    with tab1:
        render_anchor_layout(anchor_data)
    
    with tab2:
        render_force_distribution(anchor_data)
    
    with tab3:
        # render_analysis_results(anchor_data)

        # if 'analysis_results_df' in st.session_state:
        render_anchor_calculation_results(df = st.session_state["analysis_results_df"])

def render_anchor_layout(df: pd.DataFrame):
    """Render 2D anchor layout visualization"""
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
        name="Anchors"
    ))
    
    # Add centroid if multiple anchors
    if len(df) > 1:
        centroid_x = df['X'].mean()
        centroid_y = df['Y'].mean()
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
            name="Centroid"
        ))
    
    # Update layout
    fig.update_layout(
        title="Anchor Layout",
        xaxis_title="X (in)",
        yaxis_title="Y (in)",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            scaleanchor="x",  # This ensures equal aspect ratio
            scaleratio=1      # 1:1 ratio
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
# def render_anchor_layout(df: pd.DataFrame):
#     """Render 2D anchor layout visualization"""
#     fig = go.Figure()
    
#     # Add anchor points
#     fig.add_trace(go.Scatter(
#         x=df['X'],
#         y=df['Y'],
#         mode='markers+text',
#         marker=dict(
#             size=15,
#             color='blue',
#             symbol='circle',
#             line=dict(width=2, color='darkblue')
#         ),
#         text=[f"A{i+1}" for i in range(len(df))],
#         textposition="top right",
#         name="Anchors"
#     ))
    
#     # Add centroid if multiple anchors
#     if len(df) > 1:
#         centroid_x = df['X'].mean()
#         centroid_y = df['Y'].mean()
#         fig.add_trace(go.Scatter(
#             x=[centroid_x],
#             y=[centroid_y],
#             mode='markers',
#             marker=dict(
#                 size=10,
#                 color='red',
#                 symbol='x',
#                 line=dict(width=2, color='darkred')
#             ),
#             name="Centroid"
#         ))
    
#     # Update layout
#     fig.update_layout(
#         title="Anchor Layout",
#         xaxis_title="X (in)",
#         yaxis_title="Y (in)",
#         showlegend=True,
#         hovermode='closest',
#         aspectratio=dict(x=1, y=1),
#         xaxis=dict(
#             showgrid=True,
#             gridwidth=1,
#             gridcolor='LightGray',
#             zeroline=True,
#             zerolinewidth=2,
#             zerolinecolor='black'
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridwidth=1,
#             gridcolor='LightGray',
#             zeroline=True,
#             zerolinewidth=2,
#             zerolinecolor='black'
#         )
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

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

def render_project_designs_table():
    """Render the project designs summary table"""

    designs: list = st.session_state.get("data_column", [])
    if not designs:
        st.info("No designs saved yet.")
        return

    # Convert to DataFrame and transpose for better viewing
    df = pd.DataFrame(designs)

    # Reorder columns if they exist
    existing_cols = [col for col in PROJECT_DESIGN_COLUMNS if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + remaining_cols]

    # Display options
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        show_transposed = st.checkbox("Transpose Table", value=True)
    with col2:
        show_index = st.checkbox("Show Index", value=True)

    # Display table
    if show_transposed:
        st.dataframe(
            df.T,
            height=843,
            use_container_width=True
        )
    else:
        st.dataframe(
            df,
            height=400,
            use_container_width=True,
            hide_index=not show_index
        )
    
    # Export options
    st.markdown("### Export Options")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="anchor_designs.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("Clear All Designs"):
            if st.checkbox("Confirm clear all designs"):
                st.session_state.data_column = []
                st.session_state.data_column_counter = 0
                st.rerun()