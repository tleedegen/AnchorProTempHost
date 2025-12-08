import streamlit as st
import plotly.graph_objects as go
import numpy as np
from anchor_pro.streamlit.core_functions.parameters import ParameterGroup, Parameters

def plot_anchor_geometry(param: Parameters) -> go.Figure:
    """Generates a Plotly figure for the given anchor configuration."""
    

    # Fixture Geometry (Centered at 0,0)
    w, h = param.Bx, param.By
    x_min, x_max = -w / 2, w / 2
    y_min, y_max = -h / 2, h / 2
    
    fig = go.Figure()

    # 1. Fixture Boundary
    # fig.add_shape(
    #     type="rect",
    #     x0=x_min, y0=y_min, x1=x_max, y1=y_max,
    #     line=dict(color="RoyalBlue", width=2),
    #     fillcolor="LightSkyBlue",
    #     opacity=0.3,
    #     layer="below"
    # )
    # Add a dummy trace for the legend
    # fig.add_trace(go.Scatter(
    #     x=[None], y=[None],
    #     mode='markers',
    #     marker=dict(color='LightSkyBlue', symbol='square'),
    #     name='Fixture'
    # ))
    
    # 2. Anchors
    anchors = param.xy_anchors
    if anchors is not None and len(anchors) > 0:
        fig.add_trace(go.Scatter(
            x=anchors[:, 0],
            y=anchors[:, 1],
            mode='markers',
            marker=dict(size=12, color='Blue', symbol='circle', line=dict(width=1, color='White')),
            name='Anchors'
        ))

    # 3. Concrete Edges
    # Edge logic: cx_neg is distance from Left Fixture Edge to Concrete Edge
    
    # Collect key points to set range
    x_coords = [x_min, x_max]
    y_coords = [y_min, y_max]
    if anchors is not None and len(anchors) > 0:
        x_coords.extend(anchors[:, 0])
        y_coords.extend(anchors[:, 1])

    # Calculate Edge Coordinates if finite
    # Edges relative to origin = (Fixture Edge) +/- (Edge Distance)
    edge_x_min = (x_min - param.cx_neg) if not np.isinf(param.cx_neg) else None
    edge_x_max = (x_max + param.cx_pos) if not np.isinf(param.cx_pos) else None
    edge_y_min = (y_min - param.cy_neg) if not np.isinf(param.cy_neg) else None
    edge_y_max = (y_max + param.cy_pos) if not np.isinf(param.cy_pos) else None

    if edge_x_min is not None: x_coords.append(edge_x_min)
    if edge_x_max is not None: x_coords.append(edge_x_max)
    if edge_y_min is not None: y_coords.append(edge_y_min)
    if edge_y_max is not None: y_coords.append(edge_y_max)

    # Determine view margins
    xm, xM = min(x_coords), max(x_coords)
    ym, yM = min(y_coords), max(y_coords)
    
    span_x = max(xM - xm, 10.0)
    span_y = max(yM - ym, 10.0)
    margin_x = span_x * 0.2
    margin_y = span_y * 0.2
    
    view_x_min, view_x_max = xm - margin_x, xM + margin_x
    view_y_min, view_y_max = ym - margin_y, yM + margin_y

    # Draw Edges
    line_style = dict(color="Gray", width=2, dash="dashdot")
    
    # Left Edge (-X)
    if edge_x_min is not None:
        fig.add_shape(type="line", x0=edge_x_min, y0=view_y_min, x1=edge_x_min, y1=view_y_max, line=line_style)
        fig.add_annotation(x=edge_x_min, y=(view_y_min+view_y_max)/2, text="Conc. Edge", textangle=-90, showarrow=False, xshift=-15)

    # Right Edge (+X)
    if edge_x_max is not None:
        fig.add_shape(type="line", x0=edge_x_max, y0=view_y_min, x1=edge_x_max, y1=view_y_max, line=line_style)
        fig.add_annotation(x=edge_x_max, y=(view_y_min+view_y_max)/2, text="Conc. Edge", textangle=-90, showarrow=False, xshift=15)

    # Bottom Edge (-Y)
    if edge_y_min is not None:
        fig.add_shape(type="line", x0=view_x_min, y0=edge_y_min, x1=view_x_max, y1=edge_y_min, line=line_style)
        fig.add_annotation(x=(view_x_min+view_x_max)/2, y=edge_y_min, text="Conc. Edge", showarrow=False, yshift=-15)

    # Top Edge (+Y)
    if edge_y_max is not None:
        fig.add_shape(type="line", x0=view_x_min, y0=edge_y_max, x1=view_x_max, y1=edge_y_max, line=line_style)
        fig.add_annotation(x=(view_x_min+view_x_max)/2, y=edge_y_max, text="Conc. Edge", showarrow=False, yshift=15)

    # Layout config
    fig.update_layout(
        title=dict(text=f"Geometry Preview: {param.name}", x=0.5, xanchor='center'),
        xaxis_title="X (in)",
        yaxis_title="Y (in)",
        xaxis=dict(range=[view_x_min, view_x_max], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[view_y_min, view_y_max]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        width=500,
        height=500
    )
    
    return fig

def render_anchor_geometry_preview():
    """
    Renders the preview section including a select box for parameters and the Plotly chart.
    """
    st.markdown("#### Preview")

    group: ParameterGroup = st.session_state.get('parameter_group')

    
    names = group.get_names()
    if not names:
        return

    # Use index 0 as default
    selected_name = st.selectbox(
        "Select Parameter Set to Visualize", 
        options=names, 
        key="preview_config_select"
    )
    
    # Display Plot
    if selected_name:
        selected_param = group.get_parameter_by_name(selected_name)
        if selected_param:
            fig = plot_anchor_geometry(selected_param)
            st.plotly_chart(fig, width='stretch')