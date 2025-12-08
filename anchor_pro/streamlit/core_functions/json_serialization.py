import json
import logging
import numpy as np
from dataclasses import asdict
from typing import List, Dict, Any, Optional

# Import your data structures
from anchor_pro.streamlit.core_functions.parameters import Parameters, ParameterGroup
from anchor_pro.elements.concrete_anchors import Profiles, AnchorPosition

# Configure a logger for serialization errors
logger = logging.getLogger(__name__)

class AnchorProJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle NumPy arrays and Enums for AnchorPro parameters.
    """
    def default(self, obj):
        # Convert NumPy arrays to standard lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Convert Enum members to their values (strings)
        if isinstance(obj, (Profiles, AnchorPosition)):
            return obj.value
        return super().default(obj)

def to_json_str(parameter_group: ParameterGroup, indent: int = 4) -> str:
    """
    Serializes a ParameterGroup into a JSON formatted string.
    """
    data = asdict(parameter_group)
    return json.dumps(data, cls=AnchorProJSONEncoder, indent=indent)

def from_json_str(json_str: str) -> ParameterGroup:
    """
    Deserializes a JSON string into a ParameterGroup object.
    Restores NumPy arrays and Enum objects from the basic JSON types.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file format: {e}")
    
    # Handle structure variations
    raw_params = []
    if isinstance(data, dict):
        raw_params = data.get("parameters", [])
        # Handle case where a single parameter object is passed directly
        if not raw_params and "name" in data:
            raw_params = [data]
    elif isinstance(data, list):
        raw_params = data
    
    restored_parameters = []

    for p_data in raw_params:
        try:
            # 1. Restore Enums
            if "profile" in p_data:
                p_data["profile"] = Profiles(p_data["profile"])
            
            if "anchor_position" in p_data:
                p_data["anchor_position"] = AnchorPosition(p_data["anchor_position"])

            # 2. Restore NumPy arrays
            if "xy_anchors" in p_data:
                p_data["xy_anchors"] = np.array(p_data["xy_anchors"])
                
            if "individual_forces" in p_data and p_data["individual_forces"] is not None:
                p_data["individual_forces"] = np.array(p_data["individual_forces"])
            
            # 3. Construct Parameters object
            param = Parameters(**p_data)
            restored_parameters.append(param)
            
        except (ValueError, TypeError, KeyError) as e:
            # Log the error instead of using st.warning
            logger.warning(f"Skipping parameter set '{p_data.get('name', 'Unknown')}' due to data error: {e}")
            continue

    return ParameterGroup(parameters=restored_parameters)