from typing import Optional
import streamlit as st
from models.anchor_design_data import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams
from utils.data_loader import load_anchor_data, get_manufacturers, get_anchor_products, get_product_groups
from utils.session_state import save_design_to_session, update_active_design
from components.geometry_data_editor import render_data_editor
import pandas as pd


def render_sidebar():
    """Render the sidebar with all input fields"""
    with st.sidebar:
        st.header("Design Editor")

        # Initialize design parameters
        substrate: SubstrateParams = render_substrate_section()
        anchor_product: AnchorProduct = render_anchor_product_section()

        loading: LoadingParams
        anchor_data: pd.DataFrame
        loading, anchor_data = render_anchor_loading_section()

        installation: InstallationParams = render_installation_section()

        # Create design parameters object
        params = DesignParameters(
            substrate=substrate,
            anchor_product=anchor_product,
            loading=loading,
            installation=installation
        )

        # Record button
        if st.button("Record Data", type="primary", use_container_width=True):
            save_design_to_session(params)

        if st.button('Recalculate Design', type="secondary", use_container_width=True):
            update_active_design(params)
            st.success("Design recalculated!")

        return params, anchor_data


def render_substrate_section() -> SubstrateParams:
    """Render substrate input fields"""
    with st.expander("Substrate Parameters", expanded=True):
        lw_factor: Optional[float] = None

        st.subheader("Substrate")
        substrate_params = SubstrateParams()
        col1, col2, col3 = st.columns(3)

        with col1:
            base_material = st.selectbox(
                label = substrate_params.SUBSTRATE_FIELDS["base_material"]["label"],
                options = substrate_params.SUBSTRATE_FIELDS["base_material"]["options"],
                key = substrate_params.SUBSTRATE_FIELDS["base_material"]["key"])

        with col2:
            cracked_concrete = st.selectbox(
                label = substrate_params.SUBSTRATE_FIELDS["cracked_concrete"]["label"],
                options = substrate_params.SUBSTRATE_FIELDS["cracked_concrete"]["options"],
                index = substrate_params.SUBSTRATE_FIELDS["cracked_concrete"]["index"],
                key = substrate_params.SUBSTRATE_FIELDS["cracked_concrete"]["key"]
            )
            # CMU Not imlemented yet in backend
            grouted = st.selectbox(
                label = substrate_params.SUBSTRATE_FIELDS["grouted"]["label"],
                options = substrate_params.SUBSTRATE_FIELDS["grouted"]["options"],
                placeholder = substrate_params.SUBSTRATE_FIELDS["grouted"]["placeholder"],
                key = substrate_params.SUBSTRATE_FIELDS["grouted"]["key"])

        with col3:
            weight_class = st.selectbox(
                label = substrate_params.SUBSTRATE_FIELDS["weight_class"]["label"],
                options = substrate_params.SUBSTRATE_FIELDS["weight_class"]["options"],
                index = substrate_params.SUBSTRATE_FIELDS["weight_class"]["index"],
                key = substrate_params.SUBSTRATE_FIELDS["weight_class"]["key"])
            lw_factor = substrate_params.weight_class_lambda(weight_class)
            st.session_state['lw_factor'] = lw_factor
            st.header(lw_factor)
        with col1:
            poisson = st.number_input(
                label = substrate_params.SUBSTRATE_FIELDS["poisson"]["label"],
                min_value = substrate_params.SUBSTRATE_FIELDS["poisson"]["min_value"],
                max_value = substrate_params.SUBSTRATE_FIELDS["poisson"]["max_value"],
                value = substrate_params.SUBSTRATE_FIELDS["poisson"]["value"],
                key = substrate_params.SUBSTRATE_FIELDS["poisson"]["key"])
            
        with col2:
            concrete_thickness = st.number_input(
                label = substrate_params.SUBSTRATE_FIELDS["concrete_thickness"]["label"],
                min_value = substrate_params.SUBSTRATE_FIELDS["concrete_thickness"]["min_value"],
                value = substrate_params.SUBSTRATE_FIELDS["concrete_thickness"]["value"],
                key = substrate_params.SUBSTRATE_FIELDS["concrete_thickness"]["key"])
            
        with col3:
            concrete_profile = st.selectbox(
                label = substrate_params.SUBSTRATE_FIELDS["concrete_profile"]["label"],
                options = substrate_params.SUBSTRATE_FIELDS["concrete_profile"]["options"],
                index = substrate_params.SUBSTRATE_FIELDS["concrete_profile"]["index"],
                key = substrate_params.SUBSTRATE_FIELDS["concrete_profile"]["key"])
            
        with col1:
            edge_dist_x_neg = st.number_input(
                label = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_neg"]["label"],
                min_value = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_neg"]["min_value"],
                value = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_neg"]["value"],
                key = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_neg"]["key"])
            
        with col2:
            edge_dist_x_pos = st.number_input(
                label = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_pos"]["label"],
                min_value = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_pos"]["min_value"],
                value = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_pos"]["value"],
                key = substrate_params.SUBSTRATE_FIELDS["edge_dist_x_pos"]["key"])
            
        with col1:
            edge_dist_y_neg = st.number_input(
                label = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_neg"]["label"],
                min_value = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_neg"]["min_value"],
                value = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_neg"]["value"],
                key = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_neg"]["key"])
            
        with col2:
            edge_dist_y_pos = st.number_input(
                label = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_pos"]["label"],
                min_value = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_pos"]["min_value"],
                value = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_pos"]["value"],
                key = substrate_params.SUBSTRATE_FIELDS["edge_dist_y_pos"]["key"])
        
        anchor_position = st.selectbox(
            label = substrate_params.SUBSTRATE_FIELDS["anchor_position"]["label"],
            options = substrate_params.SUBSTRATE_FIELDS["anchor_position"]["options"],
            placeholder = substrate_params.SUBSTRATE_FIELDS["anchor_position"]["placeholder"],
            key = substrate_params.SUBSTRATE_FIELDS["anchor_position"]["key"])

        deck_location = st.selectbox(
            label = substrate_params.SUBSTRATE_FIELDS["deck_location"]["label"],
            options = substrate_params.SUBSTRATE_FIELDS["deck_location"]["options"],
            placeholder = substrate_params.SUBSTRATE_FIELDS["deck_location"]["placeholder"],
            key = substrate_params.SUBSTRATE_FIELDS["deck_location"]["key"])

        hole_diameter = st.number_input(
            label = substrate_params.SUBSTRATE_FIELDS["hole_diameter"]["label"],
            min_value = substrate_params.SUBSTRATE_FIELDS["hole_diameter"]["min_value"],
            value=substrate_params.SUBSTRATE_FIELDS['hole_diameter']['value'],
            placeholder = substrate_params.SUBSTRATE_FIELDS["hole_diameter"]["placeholder"],
            key = substrate_params.SUBSTRATE_FIELDS["hole_diameter"]["key"])

        face_side = st.selectbox(
            label = substrate_params.SUBSTRATE_FIELDS["face_side"]["label"],
            options = substrate_params.SUBSTRATE_FIELDS["face_side"]["options"],
            placeholder = substrate_params.SUBSTRATE_FIELDS["face_side"]["placeholder"],
            key = substrate_params.SUBSTRATE_FIELDS["face_side"]["key"])

        return SubstrateParams(
            fc=base_material,
            weight_classification_base=weight_class,
            poisson=poisson,
            t_slab=concrete_thickness,
            deck_location=deck_location,
            cracked_concrete=cracked_concrete,
            cx_neg=edge_dist_x_neg,
            cx_pos=edge_dist_x_pos,
            cy_neg=edge_dist_y_neg,
            cy_pos=edge_dist_y_pos,
            profile=concrete_profile,
            anchor_position=anchor_position,
            lw_factor=lw_factor,
            grouted = grouted == "Yes" if grouted else None,
            hole_diameter = hole_diameter,
            face_side = face_side,
        )


def render_anchor_product_section() -> AnchorProduct:
    """Render anchor product selection fields"""
    with st.expander("Anchor Product", expanded=True):
        st.subheader("Anchor Product")

        anchor_product = AnchorProduct()

        manufacturer = st.selectbox(
            label = anchor_product.ST_ANCHOR_FIELDS["manufacturer"]["label"],
            options = anchor_product.ST_ANCHOR_FIELDS["manufacturer"]["options"],
            placeholder = anchor_product.ST_ANCHOR_FIELDS["manufacturer"]["placeholder"],
            key = anchor_product.ST_ANCHOR_FIELDS["manufacturer"]["key"]
        )

        # Filter products based on manufacturer if selected
        if manufacturer:
            anchor_products = get_anchor_products(anchor_product.anchor_parameters, 
                                                manufacturer = manufacturer)
        else:
            anchor_products = get_anchor_products(anchor_product.anchor_parameters)

        # Specified product selection
        specified_product = st.selectbox(
            label = anchor_product.ST_ANCHOR_FIELDS["specified_product"]["label"],
            #TODO: Could be a cleaner way to implement options with anchor_products.SUBSTRATE_FIELDS
            options = [None] + list(anchor_products),
            placeholder = anchor_product.ST_ANCHOR_FIELDS["specified_product"]["placeholder"],
            key = anchor_product.ST_ANCHOR_FIELDS["specified_product"]["key"],
            index = anchor_product.ST_ANCHOR_FIELDS["specified_product"]["index"]
        )

        # Product group selection
        #TODO: Implement product group
        # product_groups = get_product_groups(anchor_product.anchor_parameters)


        # product_group = st.selectbox(
        #     "Product Group",
        #     options=[None] + list(product_groups),
        #     index=0,
        #     placeholder="Select...",
        #     key="product_group"
        # )
        
        return AnchorProduct(
            mode=manufacturer,
            specified_product=specified_product,
            # product_group=product_group
        )


def render_anchor_loading_section() -> tuple[LoadingParams, pd.DataFrame]:
    """Render anchor loading input fields"""
    with st.expander("Anchor Loading", expanded=True):
        st.subheader("Anchor Loading")

        loading_params = LoadingParams()
        
        load_location = st.selectbox(
            loading_params.ST_LOADING_FIELDS['load_location']['label'],
            options=loading_params.ST_LOADING_FIELDS['load_location']['options'],
            index=loading_params.ST_LOADING_FIELDS['load_location']['index'],
            placeholder=loading_params.ST_LOADING_FIELDS['load_location']['placeholder'],
            key=loading_params.ST_LOADING_FIELDS['load_location']['key']
        )
        
        anchor_data = render_data_editor()
        
        st.markdown("**Moments:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mx = st.number_input(
                "Mx (lb-in)",
                value=None,
                placeholder="0.0",
                key="mx"
            )
        
        with col2:
            my = st.number_input(
                "My (lb-in)",
                value=None,
                placeholder="0.0",
                key="my"
            )
        
        with col3:
            t = st.number_input(
                "Mz (lb-in)",
                value=None,
                placeholder="0.0",
                key="t"
            )

        st.markdown("**Options:**")
        col1, col2 = st.columns(2)
        
        with col1:
            seismic = st.selectbox(
                label=loading_params.ST_LOADING_FIELDS['seismic']['label'],
                options=loading_params.ST_LOADING_FIELDS['seismic']['options'],
                index=loading_params.ST_LOADING_FIELDS['seismic']['index'],
                format_func=lambda x: "Yes" if x else "No",
                key=loading_params.ST_LOADING_FIELDS['seismic']['key']
            )
        
        with col2:
            phi_override = st.selectbox(
                label=loading_params.ST_LOADING_FIELDS['phi_override']['label'],
                options=loading_params.ST_LOADING_FIELDS['phi_override']['options'],
                index= loading_params.ST_LOADING_FIELDS['phi_override']['index'],
                format_func=lambda x: "Yes" if x else "No",
                key=loading_params.ST_LOADING_FIELDS['phi_override']['key'] 
            )
        
        return LoadingParams(
            location=load_location,
            mx=mx or 0.0,
            my=my or 0.0,
            t=t or 0.0,
            seismic=seismic,
            phi_override=phi_override
        ), anchor_data

def render_installation_section() -> InstallationParams:
    """Render installation conditions fields"""
    with st.expander('Installation Conditions', expanded=True):
        st.header("Installation Conditions")
        
        # First row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hef = st.number_input(
                "hef (in)",
                min_value=0.0,
                value=None,
                placeholder="Input...",
                key="hef"
            )
            
            drilling_type = st.selectbox(
                "Drilling Type",
                options=[
                    None,
                    "Hammer Drill",
                    "Core Drill",
                    "Diamond Core",
                    "Rotary Impact"
                ],
                index=0,
                placeholder="Select...",
                key="drilling_type"
            )
        
        with col2:
            short_term_temp = st.number_input(
                "Short Term Temp (°F)",
                value=None,
                placeholder="Input...",
                key="short_term_temp"
            )
            
            inspection_condition = st.selectbox(
                "Inspection Condition",
                options=[
                    None,
                    "Continuous",
                    "Periodic",
                    "None"
                ],
                index=0,
                placeholder="Select...",
                key="inspection_condition"
            )
        
        with col3:
            long_term_temp = st.number_input(
                "Long Term Temp (°F)",
                value=None,
                placeholder="Input...",
                key="long_term_temp"
            )
            
            moisture_condition = st.selectbox(
                "Moisture Condition",
                options=[
                    None,
                    "Dry",
                    "Water-Saturated",
                    "Water-Filled",
                    "Submerged"
                ],
                index=0,
                placeholder="Select...",
                key="moisture_condition"
            )
        
        return InstallationParams(
            hef=hef,
            short_term_temp=short_term_temp,
            long_term_temp=long_term_temp,
            drilling_type=drilling_type,
            inspection_condition=inspection_condition,
            moisture_condition=moisture_condition,
        )

def render_sidebar_footer():
    """Render footer information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
