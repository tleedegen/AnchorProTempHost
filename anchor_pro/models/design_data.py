"""Stores and manages data used with design editor"""
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, List, ClassVar
import streamlit as st
from utils.data_loader import load_anchor_data, get_manufacturers, get_anchor_products, get_product_groups
import pandas as pd

@dataclass
class SubstrateParams:
    """Stores substrate parameters for user editing"""
    base_material: Optional[str] = None
    weight_class: Optional[str] = None
    poisson: Optional[float] = None
    concrete_thickness: Optional[float] = None
    edge_dist_x_neg: Optional[float] = None
    edge_dist_x_pos: Optional[float] = None
    edge_dist_y_neg: Optional[float] = None
    edge_dist_y_pos: Optional[float] = None
    concrete_profile: Optional[str] = None
    anchor_position: Optional[str] = None

    grouted: Optional[bool] = None
    deck_location: Optional[str] = None
    hole_diameter: Optional[float] = None
    face_side: Optional[str] = None
    cracked_concrete: bool = True
    #Default Concrete Parameter


    SUBSTRATE_FIELDS = {
        "base_material":{
            "label": "Base Material",
            "options": (2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 8500),
            "key": "fc"
        },
        "weight_class":{
            "label": "NWC / LWC",
            "options": ("NWC", "LWC"),
            "index": 0,
            "key": "weight_classification_base"
        },
        "poisson":{
            "label": "poisson",
            "min_value": 0.0,
            "max_value": 1.0,
            "value": 0.2,
            "key": "poisson"
        },
        "concrete_thickness":{
            "label": "Concrete Thickness(in)",
            "min_value": 0.0,
            "value": 12.0,
            "key": "t_slab"
        },
        "concrete_profile":{
            "label": "Slab / Filled Deck",
            "options": ("Slab", "Filled Deck"),
            "index": 0,
            "key": "profile"
        },
        "edge_dist_x_neg":{
            "label": "Edge dist -x",
            "min_value": 0.0,
            "value": 12.0,
            "key": "cx_neg"
        },
        "edge_dist_x_pos":{
            "label": "Edge dist +x",
            "min_value": 0.0,
            "value": 12.0,
            "key": "cx_pos"
        },
        "edge_dist_y_neg":{
            "label": "Edge dist -y",
            "min_value": 0.0,
            "value": 12.0,
            "key": "cy_neg"
        },
        "edge_dist_y_pos":{
            "label": "Edge dist +y",
            "min_value": 0.0,
            "value": 12.0,
            "key": "cy_pos"
        },
        "Bx":{
            "label": "Bounding Box x",
            "min_value": 0.0,
            "value": 10.0,
            "key": "Bx"
        },
        "By":{
            "label": "Bounding Box y",
            "min_value": 0.0,
            "value": 10.0,
            "key": "By"
        },
        "grouted":{
            "label": "Grouted / Not-grouted",
            "options": ("Grouted", "Not-grouted"),
            "key": "grouted",
            "placeholder": "Select..."
        },
        "cracked_concrete":{
            "label": "Cracked / Uncracked",
            "options": ("Cracked", "Uncracked"),
            "index": 0,
            "key": "cracked_concrete"
        },
        "anchor_position":{
            "label": "Anchor Position",
            "options": ('top', 'soffit'),
            "key": "anchor_position",
            "placeholder": "Select..."
        },
        "deck_location":{
            "label": "Deck Installation Location",
            "options": ("Top", "Upper Flute", "Lower Flute"),
            "key": "deck_location",
            "placeholder": "Select..."
        },
        "hole_diameter":{
            "label": "Hole Diameter of Fastened Part",
            "min_value": 0.0,
            "placeholder": "Input...",
            "key": "hole_diameter"
        },
        "face_side":{
            "label": "Face, Side",
            "options": (None, "Face", "Side", "Top"),
            "placeholder": "Select...",
            "key": "face_side"
        }
    }

    def weight_class_lambda(self, weight_class: str):
        """Determine lambda given concrete weight classification"""
        if weight_class is "LWC":
            return 0.75
        else: return 1.0

@dataclass
class AnchorProduct:
    """Stores anchor products for user editing"""
    mode: Optional[str] = None
    specified_product: Optional[str] = None
    product_group: Optional[str] = None
    anchor_parameters: pd.DataFrame = field(default_factory = load_anchor_data)

    def __post_init__(self):
        self.anchor_manufacturer: tuple = get_manufacturers(self.anchor_parameters)

        self.SUBSTRATE_FIELDS = {
            "manufacturer": {
                "label": "Manufacturer",
                "options": self.anchor_manufacturer,
                "placeholder": "Select...",
                "key": "anchor_product_mode"
            },
            "specified_product": {
                "label": "Specified Product",
                "placeholder": "Select...",
                "key": "specified_product",
                "index": 1
            }
        }



@dataclass
class LoadingParams:
    """Stores loading parameters for user editing"""
    location: Optional[str] = None
    vx: float = 0.0
    vy: float = 0.0
    n: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    t: float = 0.0
    seismic: bool = False
    phi_override: bool = False

@dataclass
class InstallationParams:
    """Stores installation parameters for user editing"""
    hef: Optional[float] = None
    short_term_temp: Optional[float] = None
    long_term_temp: Optional[float] = None
    drilling_type: Optional[str] = None
    inspection_condition: Optional[str] = None
    moisture_condition: Optional[str] = None

@dataclass
class DesignParameters:
    """Stores design editor data used in the sidebar"""
    substrate: SubstrateParams = field(default_factory = SubstrateParams)
    anchor_product: AnchorProduct = field(default_factory = AnchorProduct)
    loading: LoadingParams = field(default_factory = LoadingParams)
    installation: InstallationParams = field(default_factory = InstallationParams)
    parameters: list = field(default_factory = list)
    combined_dict: Optional[dict] = None


    def collect_parameter_names(self) -> None:
        """Collects all attributes that can be modified by user within the sidebar"""
        self.parameters = (
            [key.name for key in fields(self.substrate)] +
            [key.name for key in fields(self.anchor_product)] +
            [key.name for key in fields(self.loading)] +
            [key.name for key in fields(self.installation)])

    def parameters_to_dict(self) -> None:
        """Converts editor attributes to dict"""

        substrate_dict = asdict(self.substrate)
        anchor_product_dict = asdict(self.anchor_product)
        loading_dict = asdict(self.loading)
        installation_dict = asdict(self.installation)

        self.combined_dict = substrate_dict | anchor_product_dict | loading_dict | installation_dict


    def __post_init__(self):
        self.collect_parameter_names()
        self.parameters_to_dict()
