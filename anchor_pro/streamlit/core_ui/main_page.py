import streamlit as st
from anchor_pro.streamlit.core_ui.parameter_display import render_parameters_display
from anchor_pro.streamlit.core_ui.dcr_summary import render_dcr_summary
from anchor_pro.streamlit.core_ui.anchor_results_analysis import render_anchor_results_analysis
from anchor_pro.streamlit.core_ui.download_upload_json import render_download_upload_ui
from anchor_pro.streamlit.core_ui.anchor_geometry_preview import render_anchor_geometry_preview
from anchor_pro.streamlit.core_ui.report_generation import render_report_section


def render_main_page():
    """Render the main page of the AnchorPro app"""
    st.image('https://degenkolb.com/wp-content/uploads/Degenkolb-wh-logo.svg')
    st.title("AnchorPro Concrete")
    st.set_page_config(layout="wide")

    render_parameters_display()

    dcr_container = st.container()
    upload_container = st.container()
    with upload_container:
        render_download_upload_ui()
    with dcr_container:
        render_dcr_summary()
    render_report_section()
    col1, col2 = st.columns([1, 1])
    with col1:
        render_anchor_geometry_preview()
    with col2:
        render_anchor_results_analysis()