import streamlit as st
from core_functions.visualizations import render_visualizations, render_project_designs_table
from core_functions.save_project import render_save_load_section
from utils.session_state import app_setup, update_active_design, initialize_default_data_column
from utils.data_loader import anchor_pro_set_data
from auth.login_ui import render_login_sidebar
from auth.simple_auth import ensure_login
from core_ui.sidebar import render_sidebar



st.set_page_config(layout="wide")


def main():
    """Main function to run the AnchorPro app"""
    # Authentication check
    ensure_login()

    if "data_column" not in st.session_state:
        st.session_state['data_column'] = initialize_default_data_column()

    st.image('https://degenkolb.com/wp-content/uploads/Degenkolb-wh-logo.svg')
    st.title("AnchorPro Concrete")


    # Render sidebar
    render_sidebar()
    anchor_pro_set_data(st.session_state['data_column'])
    render_visualizations(st.session_state['data_column'][0]['anchor_geometry_forces'])

    render_save_load_section()

    # Project designs table
    st.header("All Project Designs")

    render_project_designs_table()

if __name__ == "__main__":
    main()
