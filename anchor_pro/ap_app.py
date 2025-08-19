import streamlit as st
from components.sidebar import render_sidebar
from components.geometry_data_editor import render_data_editor
from components.visualizations import render_visualizations, render_project_designs_table
from components.save_project import render_save_load_section
from utils.session_state import app_setup, update_active_design, initialize_default_data_column
from utils.data_loader import anchor_pro_set_data
from auth.login_ui import render_login_sidebar
from auth.simple_auth import ensure_login



st.set_page_config(layout="wide")


def main():
    if "my_counter" not in st.session_state:
        # Initialize a counter in session state
        st.session_state['my_counter'] = 0
    # Authentication check
    ensure_login()
    render_login_sidebar()
    st.write(st.session_state)

    if "data_column" not in st.session_state:
        st.session_state['data_column'] = initialize_default_data_column()


    st.title("AnchorPro Concrete")
    st.subheader("Design Editor")
    
    
    # Render sidebar
    design_params, anchor_data = render_sidebar()
    
    anchor_pro_set_data(st.session_state['data_column'])
    render_visualizations(anchor_data)

    render_save_load_section()

    # Project designs table
    st.header("All Project Designs")
    if st.session_state.get("data_column"):
        render_project_designs_table()
    st.write(st.session_state)
    # update_active_design(design_params)
    st.session_state['my_counter'] =  st.session_state['my_counter'] + 1
    st.header(st.session_state['my_counter'])

if __name__ == "__main__":
    main()
