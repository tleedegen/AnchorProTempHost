import streamlit as st
from components.sidebar import render_sidebar
from components.geometry_data_editor import render_data_editor
from components.visualizations import render_visualizations, render_project_designs_table
from utils.session_state import app_setup
from utils.data_loader import anchor_pro_set_data
from auth.login_ui import render_login_sidebar
from auth.simple_auth import ensure_login


st.set_page_config(layout="wide")


def main():
    ensure_login()
    render_login_sidebar()

    st.title("AnchorPro Concrete")
    st.subheader("Design Editor")
    
    # Initialize session state
    # initialize_session_state()
    
    # Render sidebar
    
    design_params = render_sidebar()
    app_setup(design_params)
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        anchor_data = render_data_editor()
    
    with col2:
        anchor_pro_set_data(st.session_state['data_column'])
        render_visualizations(anchor_data)

    # Project designs table
    st.header("All Project Designs")
    if st.session_state.get("data_column"):
        render_project_designs_table()
    st.write(st.session_state)

if __name__ == "__main__":
    main()
