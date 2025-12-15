import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
from anchor_pro.elements.concrete_anchors import Profiles, AnchorPosition

@dataclass
class Parameters:
    """Data structure holding a complete set of user inputs for an anchor design."""
    name: str
    
    # Concrete Properties
    profile: Profiles
    weight_classification: Literal["NWC", "LWC"]
    cracked_concrete: bool
    fc: float
    lw_factor: float
    t_slab: float
    poisson: float
    
    # Geometry & Layout
    xy_anchors: Any  # numpy array (n,2)
    Bx: float
    By: float
    cx_neg: float
    cx_pos: float
    cy_neg: float
    cy_pos: float
    anchor_position: AnchorPosition
    
    # Anchor Selection
    selected_anchor_id: Optional[str]
    
    # Loads
    load_mode: Literal["Global", "Individual"]
    loads: Dict[str, float]  # Used for Global Loads
    individual_forces: Optional[Any] = None # numpy array (n, 3) [N, Vx, Vy] for Individual mode

@dataclass
class ParameterGroup:
    """Manages a collection of saved Parameter sets."""
    parameters: List[Parameters] = field(default_factory=list)

    def add_parameter(self, param: Parameters):
        # Check for duplicates to avoid growing list infinitely with same name if simplistic usage
        # (Though unique names are better enforced at UI level)
        self.parameters.append(param)
    
    def update_parameter(self, param: Parameters) -> bool:
        """
        Updates an existing parameter set with the same name.
        Returns True if found and updated, False otherwise.
        """
        for i, p in enumerate(self.parameters):
            if p.name == param.name:
                self.parameters[i] = param
                return True
        return False
    
    def delete_parameter(self, name: str) -> bool:
        """
        Deletes a parameter set by name.
        Returns True if found and deleted, False otherwise.
        """
        for i, p in enumerate(self.parameters):
            if p.name == name:
                del self.parameters[i]
                return True
        return False
    
    def get_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def get_parameter_by_name(self, name: str) -> Optional[Parameters]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

def initialize_parameter_group():
    if 'parameter_group' not in st.session_state:
        st.session_state['parameter_group'] = ParameterGroup()