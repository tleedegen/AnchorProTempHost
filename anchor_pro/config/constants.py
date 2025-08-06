from dataclasses import dataclass
from typing import ClassVar
# Table Display Configuration
PROJECT_DESIGN_COLUMNS = [
    "base_material", "deck_location", "hole_diameter",
    "anchor_product_mode", "specified_product", "product_group", 
    "anchor_load_input_location", "vx", "vy", "n", "seismic_loading", 
    "phi_factor_override", "hef", "short_term_temp", "long_term_temp",
    "drilling_type", "moisture_condition", "anchor_layout_string"
]

# Layout Configuration
SIDEBAR_WIDTH = 300
MAIN_COLUMN_RATIO = [1, 1]