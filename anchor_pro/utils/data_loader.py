import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import streamlit as st
import numpy as np
import pandas as pd
from concrete_anchors import ConcreteAnchors

@st.cache_data
def load_anchor_data() -> pd.DataFrame:
    """Load anchor data from parquet file with caching"""
    try:
        file_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "anchors.parquet"
        df = pd.read_parquet(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Anchor data file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading anchor data: {str(e)}")
        return pd.DataFrame()

# @st.cache_data
def get_manufacturers(df: pd.DataFrame) -> Tuple[str, ...]:
    """Get unique manufacturers from anchor data"""
    if df.empty:
        return tuple()
    return tuple(df["manufacturer"].dropna().unique())

@st.cache_data
def get_anchor_products(df: pd.DataFrame, manufacturer: Optional[str] = None) -> Tuple[str, ...]:
    """Get anchor products, optionally filtered by manufacturer"""
    if df.empty:
        return tuple()

    if manufacturer:
        filtered_df = df[df["manufacturer"] == manufacturer]
        return tuple(filtered_df["anchor_id"].dropna().unique())
    else:
        return tuple(df["anchor_id"].dropna().unique())

@st.cache_data
def get_product_groups(df: pd.DataFrame) -> Tuple[str, ...]:
    """Get unique product groups from anchor data"""
    if df.empty:
        return tuple()
    return tuple(df["product"].dropna().unique())

def load_project_data(file_path: str) -> Dict:
    """Load saved project data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        st.error("Invalid project file format")
        return {}

def save_project_data(data: Dict, file_path: str) -> bool:
    """Save project data to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving project: {str(e)}")
        return False

def parse_anchor_layout_string(layout_string: str) -> pd.DataFrame:
    """Parse anchor layout string into DataFrame
    
    Expected format: "x1,y1;x2,y2;x3,y3"
    """
    if not layout_string:
        return pd.DataFrame(columns=['X', 'Y'])
    
    try:
        anchors = []
        for point in layout_string.split(';'):
            if ',' in point:
                x, y = point.strip().split(',')
                anchors.append({
                    'X': float(x),
                    'Y': float(y),
                    'Vx': 0.0,
                    'Vy': 0.0,
                    'N': 0.0
                })
        return pd.DataFrame(anchors)
    except Exception as e:
        st.error(f"Invalid anchor layout string format: {str(e)}")
        return pd.DataFrame(columns=['X', 'Y', 'Vx', 'Vy', 'N'])

def validate_anchor_spacing(df: pd.DataFrame, min_spacing: float = 3.0) -> List[str]:
    """Validate minimum spacing between anchors"""
    warnings = []
    
    if len(df) < 2:
        return warnings
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dx = df.iloc[i]['X'] - df.iloc[j]['X']
            dy = df.iloc[i]['Y'] - df.iloc[j]['Y']
            distance = (dx**2 + dy**2)**0.5
            
            if distance < min_spacing:
                warnings.append(
                    f"Anchors A{i+1} and A{j+1} are {distance:.2f} inches apart "
                    f"(minimum recommended: {min_spacing} inches)"
                )
    
    return warnings

def calculate_group_properties(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate anchor group geometric properties"""
    if df.empty:
        return {}
    
    # Centroid
    cx = df['X'].mean()
    cy = df['Y'].mean()
    
    # Second moments of area
    dx = df['X'] - cx
    dy = df['Y'] - cy
    
    Ix = (dy**2).sum()
    Iy = (dx**2).sum()
    Ixy = (dx * dy).sum()
    
    # Polar moment of inertia
    J = Ix + Iy
    
    return {
        'centroid_x': cx,
        'centroid_y': cy,
        'Ix': Ix,
        'Iy': Iy,
        'Ixy': Ixy,
        'J': J,
        'n_anchors': len(df)
    }


def anchor_pro_anchors(df: pd.DataFrame) -> list[list[float]]:
    """Select required anchors coordinates for anchor pro"""
    anchor_coords = df[['X', 'Y']].to_numpy().tolist()
    return anchor_coords

def anchor_pro_forces(df: pd.DataFrame):
    """Select required columns and convert to nested list"""
    anchor_pro_force = df[['N', 'Vx', 'Vy']].to_numpy()[:, np.newaxis, :]
    return anchor_pro_force

def anchor_pro_concrete_data(session_state_data_column: pd.Series) -> pd.Series:
    required_data: tuple = (
        'Bx', 'By', 'fc', 'lw_factor', 'cracked_concrete', 'poisson', 't_slab',
        'cx_neg', 'cx_pos', 'cy_neg', 'cy_pos', 'profile', 'anchor_position',
        'weight_classification_base'
    )
    anchor_pro_series = pd.Series({k: session_state_data_column[k] for k in required_data})
    return anchor_pro_series

def anchor_pro_set_data(session_state_data_column: list[pd.Series]):
    concrete_data = anchor_pro_concrete_data(session_state_data_column[0])
    xy_anchors = anchor_pro_anchors(session_state_data_column[0]["anchor_geometry_df"])
    df = load_anchor_data()
    anchor_id = session_state_data_column[0]["specified_product"]
    anchor_data = df[df['anchor_id']==anchor_id].iloc[0]

    model = ConcreteAnchors()
    model.set_data(concrete_data, xy_anchors)
    model.set_mechanical_anchor_properties(anchor_data)
    model.anchor_forces = anchor_pro_forces(st.session_state['data_column'][0]['anchor_geometry_df'])
    model.check_anchor_spacing()
    model.get_governing_anchor_group()
    model.check_anchor_capacities()

    #TODO: Check this redline later and fix with Optional
    # model.anchor_forces = anchor_pro_forces(session_state["anchor_data"])
    st.write(model.results)
    
    st.session_state['analysis_results_df'] = model.results


# Export functions for easy import
__all__ = [
    'load_anchor_data',
    'get_manufacturers',
    'get_anchor_products',
    'get_product_groups',
    'load_project_data',
    'save_project_data',
    'parse_anchor_layout_string',
    'validate_anchor_spacing',
    'calculate_group_properties'
]