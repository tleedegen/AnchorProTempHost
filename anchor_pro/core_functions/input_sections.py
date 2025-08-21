import streamlit as st
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams, Anchor, BasePlate
from utils.data_loader import  get_anchor_products
import pandas as pd


def render_substrate_section() -> SubstrateParams:
    """Render substrate input fields and records data to session_state data_column"""
    with st.expander("Substrate Parameters", expanded=True):

        st.subheader("Substrate")
        substrate_params = SubstrateParams()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.session_state['data_column'][0]['fc'] = st.selectbox(
                label = substrate_params.Fields.BaseMaterial.label,
                options = substrate_params.Fields.BaseMaterial.options,
                key = substrate_params.Fields.BaseMaterial.key,)

        with col2:
            st.session_state['data_column'][0]['cracked_concrete'] = st.selectbox(
                label = substrate_params.Fields.CrackedConcrete.label,
                options = substrate_params.Fields.CrackedConcrete.options,
                index = substrate_params.Fields.CrackedConcrete.index,
                key = substrate_params.Fields.CrackedConcrete.key
            )
            st.session_state['data_column'][0]['grouted'] = st.selectbox(
                label = substrate_params.Fields.Grouted.label,
                options = substrate_params.Fields.Grouted.options,
                placeholder = substrate_params.Fields.Grouted.placeholder,
                key = substrate_params.Fields.Grouted.key)

        with col3:
            st.session_state['data_column'][0]['weight_classification_base'] = st.selectbox(
                label = substrate_params.Fields.WeightClass.label,
                options = substrate_params.Fields.WeightClass.options,
                index = substrate_params.Fields.WeightClass.index,
                key = substrate_params.Fields.WeightClass.key,)
            st.session_state['data_column'][0]['lw_factor'] = substrate_params.weight_class_lambda(st.session_state['data_column'][0]['weight_classification_base'])

        with col1:
            st.session_state['data_column'][0]['poisson'] = st.number_input(
                label = substrate_params.Fields.Poisson.label,
                min_value = substrate_params.Fields.Poisson.min_value,
                max_value = substrate_params.Fields.Poisson.max_value,
                value = substrate_params.Fields.Poisson.value,
                key = substrate_params.Fields.Poisson.key)
            
        with col2:
            st.session_state['data_column'][0]['t_slab'] = st.number_input(
                label = substrate_params.Fields.ConcreteThickness.label,
                min_value = substrate_params.Fields.ConcreteThickness.min_value,
                value = substrate_params.Fields.ConcreteThickness.value,
                key = substrate_params.Fields.ConcreteThickness.key)
            
        with col3:
            st.session_state['data_column'][0]['profile'] = st.selectbox(
                label = substrate_params.Fields.Profile.label,
                options = substrate_params.Fields.Profile.options,
                index = substrate_params.Fields.Profile.index,
                key = substrate_params.Fields.Profile.key)
            
        with col1:
            st.session_state['data_column'][0]['cx_neg'] = st.number_input(
                label = substrate_params.Fields.EdgeDistXNeg.label,
                min_value = substrate_params.Fields.EdgeDistXNeg.min_value,
                value = substrate_params.Fields.EdgeDistXNeg.value,
                key = substrate_params.Fields.EdgeDistXNeg.key)
            
        with col2:
            st.session_state['data_column'][0]['cx_pos'] = st.number_input(
                label = substrate_params.Fields.EdgeDistXPos.label,
                min_value = substrate_params.Fields.EdgeDistXPos.min_value,
                value = substrate_params.Fields.EdgeDistXPos.value,
                key = substrate_params.Fields.EdgeDistXPos.key)
            
        with col1:
            st.session_state['data_column'][0]['cy_neg'] = st.number_input(
                label = substrate_params.Fields.EdgeDistYNeg.label,
                min_value = substrate_params.Fields.EdgeDistYNeg.min_value,
                value = substrate_params.Fields.EdgeDistYNeg.value,
                key = substrate_params.Fields.EdgeDistYNeg.key)
            
        with col2:
            st.session_state['data_column'][0]['cy_pos'] = st.number_input(
                label = substrate_params.Fields.EdgeDistYPos.label,
                min_value = substrate_params.Fields.EdgeDistYPos.min_value,
                value = substrate_params.Fields.EdgeDistYPos.value,
                key = substrate_params.Fields.EdgeDistYPos.key)
        
        st.session_state['data_column'][0]['anchor_position'] = st.selectbox(
            label = substrate_params.Fields.AnchorPosition.label,
            options = substrate_params.Fields.AnchorPosition.options,
            placeholder = substrate_params.Fields.AnchorPosition.placeholder,
            key = substrate_params.Fields.AnchorPosition.key)

        st.session_state['data_column'][0]['deck_location'] = st.selectbox(
            label = substrate_params.Fields.DeckLocation.label,
            options = substrate_params.Fields.DeckLocation.options,
            placeholder = substrate_params.Fields.DeckLocation.placeholder,
            key = substrate_params.Fields.DeckLocation.key)

        st.session_state['data_column'][0]['hole_diameter'] = st.number_input(
            label = substrate_params.Fields.HoleDiameter.label,
            min_value = substrate_params.Fields.HoleDiameter.min_value,
            value=substrate_params.Fields.HoleDiameter.value,
            placeholder = substrate_params.Fields.HoleDiameter.placeholder,
            key = substrate_params.Fields.HoleDiameter.key)

        st.session_state['data_column'][0]['face_side'] = st.selectbox(
            label = substrate_params.Fields.FaceSide.label,
            options = substrate_params.Fields.FaceSide.options,
            placeholder = substrate_params.Fields.FaceSide.placeholder,
            key = substrate_params.Fields.FaceSide.key)
        
    substrate_params = SubstrateParams(
        fc=st.session_state['data_column'][0]['fc'],
        cracked_concrete=st.session_state['data_column'][0]['cracked_concrete'],
        grouted=st.session_state['data_column'][0]['grouted'],
        lw_factor=st.session_state['data_column'][0]['lw_factor'],
        poisson=st.session_state['data_column'][0]['poisson'],
        weight_classification_base=st.session_state['data_column'][0]['weight_classification_base'],
        t_slab=st.session_state['data_column'][0]['t_slab'],
        profile=st.session_state['data_column'][0]['profile'],
        cx_neg=st.session_state['data_column'][0]['cx_neg'],
        cx_pos=st.session_state['data_column'][0]['cx_pos'],
        cy_neg=st.session_state['data_column'][0]['cy_neg'],
        cy_pos=st.session_state['data_column'][0]['cy_pos'],
        anchor_position=st.session_state['data_column'][0]['anchor_position'],
        deck_location=st.session_state['data_column'][0]['deck_location'],
        hole_diameter=st.session_state['data_column'][0]['hole_diameter'],
        face_side=st.session_state['data_column'][0]['face_side']
    )
    return substrate_params

def render_anchor_product_section() -> AnchorProduct:
    """Render anchor product selection fields and records data to sessions_state data_column"""
    with st.expander("Anchor Product", expanded=True):
        st.subheader("Anchor Product")

        anchor_product = AnchorProduct()

        manufacturer = st.selectbox(
            label = anchor_product.Fields.Manufacturer.label,
            options = anchor_product.Fields.Manufacturer.options,
            placeholder = anchor_product.Fields.Manufacturer.placeholder,
            key = anchor_product.Fields.Manufacturer.key,
        )

        # Filter products based on manufacturer if selected
        if manufacturer:
            anchor_products = get_anchor_products(anchor_product.anchor_parameters, 
                                                manufacturer = manufacturer)
        else:
            anchor_products = get_anchor_products(anchor_product.anchor_parameters)

        # Specified product selection
        st.session_state['data_column'][0]['specified_product'] = st.selectbox(
            label = anchor_product.Fields.SpecifiedProduct.label,
            options = list(anchor_products),
            placeholder = anchor_product.Fields.SpecifiedProduct.placeholder,
            key = anchor_product.Fields.SpecifiedProduct.key,
            index = anchor_product.Fields.SpecifiedProduct.index
        )

        anchor_product = AnchorProduct(
            specified_product=st.session_state['data_column'][0]['specified_product'],
        )
        return anchor_product

def render_anchor_loading_section() -> LoadingParams:
    """Render anchor loading input fields"""
    with st.expander("Anchor Loading", expanded=True):
        st.subheader("Anchor Loading")

        loading_params = LoadingParams()
        
        st.session_state['data_column'][0]['location'] = st.selectbox(
            loading_params.Fields.LoadLocation.label,
            options=loading_params.Fields.LoadLocation.options,
            index=loading_params.Fields.LoadLocation.index,
            placeholder=loading_params.Fields.LoadLocation.placeholder,
            key=loading_params.Fields.LoadLocation.key
        )
        


        st.markdown("**Options:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state['data_column'][0]['seismic'] = st.selectbox(
                label=loading_params.Fields.Seismic.label,
                options=loading_params.Fields.Seismic.options,
                index=loading_params.Fields.Seismic.index,
                format_func=lambda x: "Yes" if x else "No",
                key=loading_params.Fields.Seismic.key
            )
        
        with col2:
            st.session_state['data_column'][0]['phi_override'] = st.selectbox(
                label=loading_params.Fields.PhiOverride.label,
                options=loading_params.Fields.PhiOverride.options,
                index= loading_params.Fields.PhiOverride.index,
                format_func=lambda x: "Yes" if x else "No",
                key=loading_params.Fields.PhiOverride.key
            )
    loading_params = LoadingParams(
        location=st.session_state['data_column'][0]['location'],
        seismic=st.session_state['data_column'][0]['seismic'],
        phi_override=st.session_state['data_column'][0]['phi_override']
    )
    return loading_params

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

