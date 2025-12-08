import streamlit as st
from anchor_pro.streamlit.core_ui.sidebar import render_sidebar
from anchor_pro.streamlit.core_ui.main_page import render_main_page

def main():
    st.write(st.session_state)
    render_sidebar()
    render_main_page()
    st.write(st.session_state)


if __name__ == "__main__":
    main()