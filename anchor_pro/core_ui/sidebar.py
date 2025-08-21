import streamlit as st
from auth.login_ui import render_login_sidebar
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams, Anchor, BasePlate
from utils.session_state import save_design_to_session
from core_functions.input_sections import render_substrate_section, render_anchor_product_section, render_anchor_loading_section, render_installation_section
from core_functions.geometry_force_parameters import render_anchor_data_editor, render_baseplate_geometry
def render_sidebar():
    """Render the sidebar with all input fields"""
    with st.sidebar:
        st.header("Design Editor")

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
        #TODO: Record feature needs to be independent from UI stuff
        # Record button
        if st.button("Record Data", type="primary", use_container_width=True):
            save_design_to_session(design_params)

            st.success("Design recalculated!")
    
    render_login_sidebar()

