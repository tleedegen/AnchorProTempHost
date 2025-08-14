import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def render_anchor_calculation_results(csv_file_path: str = None, df: pd.DataFrame = None):
    """
    Render comprehensive visualization for concrete anchor calculation results.
    Can accept either a CSV file path or a DataFrame directly.
    
    Parameters:
    -----------
    csv_file_path : str, optional
        Path to the CSV file containing anchor calculation results
    df : pd.DataFrame, optional
        DataFrame containing anchor calculation results
    """
    
    # Load data
    if df is None and csv_file_path:
        df = pd.read_csv(csv_file_path)
    elif df is None:
        st.error("No data provided for visualization")
        return
    
    # Handle case where 'Limit State' might be the index
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()
    elif 'Limit State' not in df.columns:
        st.error("'Limit State' column not found in the data")
        return
    
    # Main title
    st.header("🔧 Concrete Anchor Analysis Results")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    max_utilization = df['Utilization'].max()
    governing_state = df.loc[df['Utilization'].idxmax(), 'Limit State']
    tension_dcr = df[df['Mode'] == 'Tension']['Utilization'].max() if 'Tension' in df['Mode'].values else 0
    shear_dcr = df[df['Mode'] == 'Shear']['Utilization'].max() if 'Shear' in df['Mode'].values else 0
    
    with col1:
        color = "🔴" if max_utilization > 1.0 else "🟡" if max_utilization > 0.8 else "🟢"
        st.metric(
            "Max Utilization",
            f"{max_utilization:.2f}",
            delta=f"{(1.0 - max_utilization):.2f} margin",
            delta_color="inverse"
        )
        st.caption(f"Status: {color}")
    
    with col2:
        st.metric("Governing Limit State", governing_state)
        st.caption(f"Mode: {df.loc[df['Utilization'].idxmax(), 'Mode']}")
    
    with col3:
        st.metric("Max Tension DCR", f"{tension_dcr:.2f}")
    
    with col4:
        st.metric("Max Shear DCR", f"{shear_dcr:.2f}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Utilization Summary", 
        "⚖️ Demand vs Capacity", 
        "📈 Detailed Breakdown",
        "📋 Data Table"
    ])
    
    with tab1:
        render_utilization_chart(df)
    
    with tab2:
        render_demand_capacity_comparison(df)
    
    with tab3:
        render_detailed_breakdown(df)
    
    with tab4:
        render_data_table(df)
    
    # Warning messages
    if max_utilization > 1.0:
        st.error(f"⚠️ **CAPACITY EXCEEDED**: {governing_state} has utilization of {max_utilization:.2f}")
    elif max_utilization > 0.9:
        st.warning(f"⚠️ **HIGH UTILIZATION**: {governing_state} is at {max_utilization:.1%} capacity")
    elif max_utilization > 0.8:
        st.info(f"ℹ️ Design utilization is acceptable. Maximum: {max_utilization:.1%}")
    else:
        st.success(f"✅ Design has adequate capacity. Maximum utilization: {max_utilization:.1%}")


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
        height=400,
        xaxis=dict(range=[0, max(1.2, df_sorted['Utilization'].max() * 1.1)]),
        showlegend=False,
        hovermode='y unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_demand_capacity_comparison(df: pd.DataFrame):
    """Render grouped bar chart comparing demand vs capacity"""
    
    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()
    
    # Prepare data for comparison
    comparison_data = df[['Limit State', 'Mode', 'Demand', 'Factored Capacity']].copy()
    
    # Create subplot with shared y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tension Limit States", "Shear Limit States"),
        shared_yaxes=False,
        horizontal_spacing=0.15
    )
    
    # Separate by mode
    tension_df = comparison_data[comparison_data['Mode'] == 'Tension']
    shear_df = comparison_data[comparison_data['Mode'] == 'Shear']
    
    # Tension subplot
    if not tension_df.empty:
        fig.add_trace(
            go.Bar(
                name='Demand',
                x=tension_df['Demand'],
                y=tension_df['Limit State'],
                orientation='h',
                marker_color='indianred',
                legendgroup='demand',
                showlegend=True,
                text=[f"{x:.0f}" for x in tension_df['Demand']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Factored Capacity',
                x=tension_df['Factored Capacity'],
                y=tension_df['Limit State'],
                orientation='h',
                marker_color='lightgreen',
                legendgroup='capacity',
                showlegend=True,
                text=[f"{x:.0f}" for x in tension_df['Factored Capacity']],
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # Shear subplot
    if not shear_df.empty:
        fig.add_trace(
            go.Bar(
                name='Demand',
                x=shear_df['Demand'],
                y=shear_df['Limit State'],
                orientation='h',
                marker_color='indianred',
                legendgroup='demand',
                showlegend=False,
                text=[f"{x:.0f}" for x in shear_df['Demand']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                name='Factored Capacity',
                x=shear_df['Factored Capacity'],
                y=shear_df['Limit State'],
                orientation='h',
                marker_color='lightgreen',
                legendgroup='capacity',
                showlegend=False,
                text=[f"{x:.0f}" for x in shear_df['Factored Capacity']],
                textposition='outside'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="Demand vs Factored Capacity Comparison",
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='y unified'
    )
    
    fig.update_xaxes(title_text="Force (lbs)", row=1, col=1)
    fig.update_xaxes(title_text="Force (lbs)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def render_detailed_breakdown(df: pd.DataFrame):
    """Render detailed breakdown with capacity factors"""
    
    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of modes
        mode_counts = df['Mode'].value_counts()
        fig_pie = px.pie(
            values=mode_counts.values,
            names=mode_counts.index,
            title="Limit States by Mode",
            color_discrete_map={'Tension': '#FF6B6B', 'Shear': '#4ECDC4'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Reduction factors visualization
        fig_factors = go.Figure()
        
        # Add bars for reduction factors
        fig_factors.add_trace(go.Bar(
            name='Reduction Factor (φ)',
            x=df['Limit State'],
            y=df['Reduction Factor'],
            marker_color='steelblue',
            text=[f"{x:.2f}" for x in df['Reduction Factor']],
            textposition='outside'
        ))
        
        # Add seismic factors as markers
        fig_factors.add_trace(go.Scatter(
            name='Seismic Factor',
            x=df['Limit State'],
            y=df['Seismic Factor'],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=[f"{x:.2f}" for x in df['Seismic Factor']],
            textposition='top center'
        ))
        
        fig_factors.update_layout(
            title="Safety Factors Applied",
            xaxis_title="",
            yaxis_title="Factor Value",
            height=400,
            xaxis_tickangle=-45,
            showlegend=True,
            yaxis=dict(range=[0, 1.1])
        )
        
        st.plotly_chart(fig_factors, use_container_width=True)
    
    # Interaction diagram (if both tension and shear exist)
    tension_data = df[df['Mode'] == 'Tension']
    shear_data = df[df['Mode'] == 'Shear']
    
    if not tension_data.empty and not shear_data.empty:
        st.subheader("Tension-Shear Interaction")
        
        max_tension_util = tension_data['Utilization'].max()
        max_shear_util = shear_data['Utilization'].max()
        
        # Create interaction curve (5/3 power equation)
        theta = np.linspace(0, np.pi/2, 100)
        n_ratio = np.cos(theta)
        v_ratio = np.sin(theta)
        
        fig_interaction = go.Figure()
        
        # Add interaction curve
        fig_interaction.add_trace(go.Scatter(
            x=v_ratio,
            y=n_ratio,
            mode='lines',
            name='Interaction Curve',
            line=dict(color='blue', dash='dash'),
            hovertemplate='V/Vc: %{x:.2f}<br>N/Nc: %{y:.2f}<extra></extra>'
        ))
        
        # Add current state point
        fig_interaction.add_trace(go.Scatter(
            x=[max_shear_util],
            y=[max_tension_util],
            mode='markers+text',
            name='Current Design',
            marker=dict(size=15, color='red' if (max_tension_util**(5/3) + max_shear_util**(5/3)) > 1 else 'green'),
            text=['Design Point'],
            textposition='top right',
            hovertemplate='Shear DCR: %{x:.2f}<br>Tension DCR: %{y:.2f}<extra></extra>'
        ))
        
        fig_interaction.update_layout(
            title="Tension-Shear Interaction Check",
            xaxis_title="Shear Utilization (V/Vc)",
            yaxis_title="Tension Utilization (N/Nc)",
            height=400,
            xaxis=dict(range=[0, 1.2], dtick=0.2),
            yaxis=dict(
                range=[0, 1.2], 
                dtick=0.2,
                scaleanchor="x",  # Link y-axis scale to x-axis
                scaleratio=1      # Maintain 1:1 aspect ratio
            ),
            showlegend=True
        )
        
        # Add grid
        fig_interaction.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_interaction.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_interaction, use_container_width=True)
        
        # Calculate interaction
        interaction_value = max_tension_util**(5/3) + max_shear_util**(5/3)
        st.metric(
            "Interaction Check",
            f"{interaction_value:.3f}",
            delta=f"{(1.0 - interaction_value):.3f} margin",
            delta_color="inverse"
        )
        st.caption("Per ACI 318-19 Eq. (17.10.3): (N/φNn)^(5/3) + (V/φVn)^(5/3) ≤ 1.0")


def render_data_table(df: pd.DataFrame):
    """Render the raw data table with formatting"""
    
    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()
    
    # Format the dataframe for display
    df_display = df.copy()
    
    # Apply formatting
    df_display['Demand'] = df_display['Demand'].apply(lambda x: f"{x:,.0f}")
    df_display['Nominal Capacity'] = df_display['Nominal Capacity'].apply(lambda x: f"{x:,.1f}")
    df_display['Reduction Factor'] = df_display['Reduction Factor'].apply(lambda x: f"{x:.2f}")
    df_display['Seismic Factor'] = df_display['Seismic Factor'].apply(lambda x: f"{x:.2f}")
    df_display['Factored Capacity'] = df_display['Factored Capacity'].apply(lambda x: f"{x:,.1f}")
    df_display['Utilization'] = df_display['Utilization'].apply(lambda x: f"{x:.3f}")
    
    # Display with color coding for utilization
    def highlight_utilization(val):
        try:
            num_val = float(val)
            if num_val > 1.0:
                return 'background-color: #ffcccc'
            elif num_val > 0.8:
                return 'background-color: #ffe6cc'
            else:
                return 'background-color: #ccffcc'
        except:
            return ''
    
    styled_df = df_display.style.map(
        highlight_utilization, 
        subset=['Utilization']
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="anchor_analysis_results.csv",
        mime="text/csv"
    )


# Integration function for existing app
def add_to_visualizations_tab(df_or_path):
    """
    Easy integration function to add to existing visualizations.py
    Call this function in the Analysis Results tab
    
    Note: Automatically handles DataFrames where 'Limit State' is either a column or the index.
    
    Example usage in visualizations.py:
    ```python
    with tab3:
        # This works whether results.df has 'Limit State' as index or column
        if 'analysis_results_df' in st.session_state:
            render_anchor_calculation_results(df=st.session_state.analysis_results_df)
    ```
    """
    render_anchor_calculation_results(df=df_or_path if isinstance(df_or_path, pd.DataFrame) else None,
                                     csv_file_path=df_or_path if isinstance(df_or_path, str) else None)