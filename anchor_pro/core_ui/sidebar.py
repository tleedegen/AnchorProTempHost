import streamlit as st
from auth.login_ui import render_login_sidebar
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams, Anchor, BasePlate
from utils.session_state import save_design_to_session, add_design_names_to_session
from core_functions.input_sections import render_substrate_section, render_anchor_product_section, render_anchor_loading_section, render_installation_section
from core_functions.geometry_force_parameters import render_anchor_data_editor, render_baseplate_geometry
from utils.widget_generator import copy_design_params_to_widgets

def render_sidebar():
    """Render the sidebar with all input fields"""
    with st.sidebar:
        st.header("Design Editor")

        if 'btn_copy_design' in st.session_state and 'copy_index' in st.session_state and st.session_state['btn_copy_design']:
            if isinstance(st.session_state['copy_index'], int):
                copy_design_params_to_widgets(st.session_state['copy_index'])

        # Initialize design parameters
        substrate_params: SubstrateParams = render_substrate_section()
        anchor_product: AnchorProduct = render_anchor_product_section()
        loading_params: LoadingParams = render_anchor_loading_section()
        anchor: Anchor = render_anchor_data_editor()
        baseplate: BasePlate = render_baseplate_geometry()
        install_params: InstallationParams = render_installation_section()

        design_params: DesignParameters = DesignParameters(
            substrate=substrate_params,
            anchor_product=anchor_product,
            loading=loading_params,
            anchor=anchor,
            baseplate=baseplate,
            installation=install_params
        )

        @st.dialog('Save Design')
        def design_name_dialog():
            name = st.text_input('Enter name for this design')
            if st.button('Save'):
                add_design_names_to_session(name)
                st.success(f"Design saved as '{name}'")
                save_design_to_session(design_params)

                st.success("Design saved!")
                st.rerun()

        design_index = st.selectbox("Toggle Design to Copy or Overwrite", options=range(1,len(st.session_state['data_column'])), format_func=lambda x: st.session_state['design_names'][x-1], key='copy_index')
        st.button('Copy Design Values to Widgets', type="secondary", use_container_width=True, key="btn_copy_design")
        if st.button('Overwrite Current Design', type="secondary", use_container_width=True, key="btn_overwrite_design"):
            st.session_state['data_column'][design_index] = st.session_state['data_column'][0].copy()
            st.success(f"Design {design_index} overwritten with default design.")
            st.rerun()

        #TODO: Record feature needs to be independent from UI stuff
        # Save current design snapshot


        if st.button("Save Data As...", type="primary", use_container_width=True):
            design_name_dialog()

    render_login_sidebar()

