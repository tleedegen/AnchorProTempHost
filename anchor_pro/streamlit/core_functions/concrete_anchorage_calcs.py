import pandas as pd
import numpy as np
from anchor_pro.streamlit.core_functions.parameters import Parameters
from anchor_pro.elements.concrete_anchors import (
    ConcreteAnchors, ConcreteProps, GeoProps, ConcreteAnchorResults,
    MechanicalAnchorProps, AnchorBasicInfo, Phi, AnchorTypes, InstallationMethod
)
from anchor_pro.elements.elastic_bolt_group import ElasticBoltGroupProps, calculate_bolt_group_forces

def _get(data, key, default=None):
    """Helper to safely get values from pandas Series or dict."""
    if hasattr(data, "get"):
        return data.get(key, default)
    try:
        return data.at[key]
    except KeyError:
        return default

def create_anchor_props(anchor_data: pd.Series, anchor_obj: ConcreteAnchors) -> MechanicalAnchorProps:
    """
    Factory function to create MechanicalAnchorProps from the catalog row 
    and the ConcreteAnchors object (used for interpolation based on slab thickness).
    """
    cp = anchor_obj.concrete_props
    
    # Interpolation logic for spacing/edge distance requirements based on slab thickness
    suffix = '_deck' if cp.profile.value == 'Filled Deck' else '_slab'
    
    h_vals = [float(_get(anchor_data, f"hmin1{suffix}")), float(_get(anchor_data, f"hmin2{suffix}"))]
    c1_vals = [float(_get(anchor_data, f"c11{suffix}")), float(_get(anchor_data, f"c21{suffix}"))]
    c2_vals = [float(_get(anchor_data, f"c12{suffix}")), float(_get(anchor_data, f"c22{suffix}"))]
    s1_vals = [float(_get(anchor_data, f"s11{suffix}")), float(_get(anchor_data, f"s21{suffix}"))]
    s2_vals = [float(_get(anchor_data, f"s12{suffix}")), float(_get(anchor_data, f"s22{suffix}"))]
    cac_vals = [float(_get(anchor_data, f"cac1{suffix}")), float(_get(anchor_data, f"cac2{suffix}"))]
    
    t_slab = cp.t_slab
    
    # Perform interpolation
    c1 = float(np.interp(t_slab, h_vals, c1_vals))
    s1 = float(np.interp(t_slab, h_vals, s1_vals))
    c2 = float(np.interp(t_slab, h_vals, c2_vals))
    s2 = float(np.interp(t_slab, h_vals, s2_vals))
    cac = float(np.interp(t_slab, h_vals, cac_vals))
    hmin = h_vals[0]

    # Select Cracked/Uncracked and Seismic properties
    is_cracked = cp.cracked_concrete
    kc = float(_get(anchor_data, "kc_cr")) if is_cracked else float(_get(anchor_data, "kc_uncr"))
    Np = _get(anchor_data, "Np_eq") 
    if pd.isna(Np):
         Np = _get(anchor_data, "Np_cr") if is_cracked else _get(anchor_data, "Np_uncr")
         
    K = float(_get(anchor_data, "K_cr")) if is_cracked else float(_get(anchor_data, "K_uncr"))
    
    # Phi Factors
    phi = Phi(
        saN=_get(anchor_data, 'phi_saN'),
        pN=_get(anchor_data, 'phi_pN'),
        cN=_get(anchor_data, 'phi_cN'),
        cV=_get(anchor_data, 'phi_cV'),
        saV=_get(anchor_data, 'phi_saV'),
        cpV=_get(anchor_data, 'phi_cpV'),
        eqV=_get(anchor_data, 'phi_eqV'),
        eqN=_get(anchor_data, 'phi_eqN'),
        seismic=0.75, # Default seismic factor
        aN=_get(anchor_data, 'phi_aN', None)
    )

    info = AnchorBasicInfo(
        anchor_id=_get(anchor_data, 'anchor_id'),
        installation_method=InstallationMethod.post, 
        anchor_type=AnchorTypes.expansion, 
        manufacturer=_get(anchor_data, 'manufacturer'),
        product=_get(anchor_data, 'product'),
        product_type=_get(anchor_data, 'product_type'),
        esr=str(_get(anchor_data, 'esr')),
        cost_rank=_get(anchor_data, 'cost_rank')
    )

    return MechanicalAnchorProps(
        info=info,
        fya=float(_get(anchor_data, "fya")),
        fua=float(_get(anchor_data, "fua")),
        Nsa=float(_get(anchor_data, "Nsa")),
        Np=float(Np),
        kc=kc,
        kc_uncr=float(_get(anchor_data, "kc_uncr")),
        kc_cr=float(_get(anchor_data, "kc_cr")),
        le=float(_get(anchor_data, "le")),
        da=float(_get(anchor_data, "da")),
        cac=cac,
        esr=str(_get(anchor_data, "esr")),
        hef_default=float(_get(anchor_data, "hef_default")),
        Vsa=float(_get(anchor_data, "Vsa_eq")) if not pd.isna(_get(anchor_data, "Vsa_eq")) else float(_get(anchor_data, "Vsa_default")),
        K=K,
        K_cr=float(_get(anchor_data, "K_cr")),
        K_uncr=float(_get(anchor_data, "K_uncr")),
        Kv=float(_get(anchor_data, "Kv")),
        hmin=hmin,
        c1=c1, s1=s1, c2=c2, s2=s2,
        phi=phi,
        abrg=float(_get(anchor_data, "abrg")) if not pd.isna(_get(anchor_data, "abrg")) else None
    )

def evaluate_concrete_anchors(param: Parameters, df_catalog: pd.DataFrame) -> ConcreteAnchorResults:
    """
    Main entry point for calculation.
    1. Extracts parameters.
    2. Distributes global loads to individual anchors (if applicable).
    3. Runs ACI 318 evaluation.
    """
    
    # 1. Setup Geometry and Concrete Properties
    geo_props = GeoProps(
        xy_anchors=param.xy_anchors,
        Bx=param.Bx, 
        By=param.By,
        cx_neg=param.cx_neg, cx_pos=param.cx_pos,
        cy_neg=param.cy_neg, cy_pos=param.cy_pos,
        anchor_position=param.anchor_position
    )

    concrete_props = ConcreteProps(
        weight_classification=param.weight_classification,
        profile=param.profile,
        fc=param.fc,
        lw_factor=param.lw_factor,
        cracked_concrete=param.cracked_concrete,
        poisson=param.poisson,
        t_slab=param.t_slab
    )

    # 2. Instantiate Anchor Object
    anchor_obj = ConcreteAnchors(geo_props=geo_props, concrete_props=concrete_props)
    
    # 3. Retrieve Catalog Data and Set Props
    if not param.selected_anchor_id:
        raise ValueError("No anchor selected.")
        
    anchor_row = df_catalog[df_catalog['anchor_id'] == param.selected_anchor_id].iloc[0]
    mech_props = create_anchor_props(anchor_row, anchor_obj)
    anchor_obj.set_anchor_props(mech_props)

    # 4. Determine Loads (Global vs Individual)
    
    # Check if load_mode is explicit; otherwise infer from presence of individual_forces
    use_individual = (getattr(param, 'load_mode', 'Global') == 'Individual')
    
    if use_individual and param.individual_forces is not None:
        # User defined specific loads per anchor [N, Vx, Vy]
        # Shape is (n_anchors, 3). Backend expects (n_anchors, 3, n_theta).
        # We assume n_theta = 1 for static/manual assignment.
        forces_flat = param.individual_forces # (n, 3)
        anchor_forces = forces_flat[:, :, np.newaxis] # (n, 3, 1)
        
    else:
        # Default: Global Loads -> Elastic Bolt Group Distribution
        # We use ElasticBoltGroupProps to calculate inertias and distribute global loads
        bg_props = ElasticBoltGroupProps(
            w=param.Bx,
            h=param.By,
            xy_anchors=param.xy_anchors,
            plate_centroid_XYZ=np.array([0., 0., 0.]), # Local calc, origin at center
            local_x=np.array([1., 0., 0.]),
            local_y=np.array([0., 1., 0.]),
            local_z=np.array([0., 0., 1.])
        )
        
        # Extract Loads (User Inputs)
        N_load = np.array([param.loads.get("N", 0.0)])
        Vx_load = np.array([param.loads.get("Vx", 0.0)])
        Vy_load = np.array([param.loads.get("Vy", 0.0)])
        Mx_load = np.array([param.loads.get("Mx", 0.0)])
        My_load = np.array([param.loads.get("My", 0.0)])
        T_load = np.array([param.loads.get("T", 0.0)])

        # Calculate individual anchor forces
        # Returns (n_anchor, 3, n_theta=1) -> [N, Vx, Vy]
        anchor_forces = calculate_bolt_group_forces(
            N=N_load, Vx=Vx_load, Vy=Vy_load, 
            Mx=Mx_load, My=My_load, T=T_load,
            n_anchors=bg_props.n_anchors,
            inert_c=bg_props.inert_props_cent,
            inert_x=bg_props.inert_props_x,
            inert_y=bg_props.inert_props_y
        )

    # 5. Run Evaluation
    results = anchor_obj.evaluate(anchor_forces)
    
    return results