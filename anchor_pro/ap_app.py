import streamlit as st
from components.sidebar import render_sidebar
from components.data_editor import render_data_editor
from components.visualizations import render_visualizations, render_project_designs_table
from utils.session_state import initialize_session_state
from utils.data_loader import anchor_pro_set_data

st.set_page_config(layout="wide")

def main():
    st.title("AnchorPro Concrete")
    st.subheader("Design Editor")
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    design_params = render_sidebar()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        anchor_data = render_data_editor()
    
    with col2:
        anchor_pro_set_data(st.session_state)
        render_visualizations(anchor_data)
        

    
    # Project designs table
    st.header("All Project Designs")
    if st.session_state.get("data_column"):
        render_project_designs_table()
    st.write(st.session_state)
    st.write()

if __name__ == "__main__":
    main()
