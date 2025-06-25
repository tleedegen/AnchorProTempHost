import numpy as np
from pathlib import Path
import streamlit as st
import pandas as pd
from scripts.anchor_pro_test_formulas import *

st.set_page_config(layout="wide")

st.title("AnchroPro Concrete")
st.subheader("Design Editor")

#ADD product list table
specified_product_list = []

#Create sidebar with input fields
with st.sidebar:
    st.header("Design Editor")
    st.subheader("Substraight")
    sidebar_column1, sidebar_column2 = st.columns(2)
    with sidebar_column1:
        st.selectbox("Base Material",
                    ("2000",
                    "2500", 
                    "3000", 
                    "4000", 
                    "5000", 
                    "6000", 
                    "7000", 
                    "8000", 
                    "8500",),
                    accept_new_options = True,
                    key = "base_material")

    with sidebar_column2:
        st.selectbox(label = "Grouted / Not-grouted",
                    placeholder = "Select...",
                    index = None,
                    key = "grouted/not-grouted",
                    options = ("Yes", "No"))

    st.selectbox(label = "Deck Installation Location",
                options =
                ("Top",
                "Upper Flute",
                "Lower Flute"),
                index = None,
                placeholder = "Select...",
                key = "deck_installation_location")

    st.number_input(label = "Hole Diameter of Fastened Part",
                    min_value = 0.0,
                    value = None,
                    placeholder = "Input...",
                    key = "hole_diameter_of_fastened_part")

    st.selectbox(label = "Face, Side",
                index = None,
                placeholder = "Select...",
                options =
                ("Face",
                "Side",
                "Top"),
                key = "face, side")

    st.subheader("Anchor Product")

    #Creating path to anchor parquet
    file_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "anchors.parquet"
    df = pd.read_parquet(file_path)


    #Remove duplicate manufacturers for anchor product mode options
    manufacturer_tuple = tuple(df["Manufacturer"].drop_duplicates())
    st.selectbox(label = "Anchor Product Mode",
                index = None,
                placeholder = "Select...",
                options =
                manufacturer_tuple,
                key = "anchor_product_mode")

    #Drop empty element and only add unique elements for options selection
    anchor_product_mode_options = tuple(df["Anchor ID"].dropna().unique())
    #Options should be taken from excel sheet
    st.selectbox(label = "Specified Product",
                index = None,
                placeholder = "Select...",
                options = anchor_product_mode_options,
                key = "specified_product")

    #
    product_group_options = tuple(df["Product"].drop_duplicates())
    st.selectbox(label = "Product Group",
                 index = None,
                 placeholder = "Select...",
                 key = "product_group",
                 options = product_group_options)

    st.subheader("Anchor Loading")

    st.selectbox(label = "Anchor Load Input Location",
                 key = "anchor_load_input_location",
                 placeholder = "Select...",
                 index = None,
                 options =
                 ("Individual Anchors",
                 "Group Origin"))
    sidebar_column3, sidebar_column4, sidebar_column5, sidebar_column6 = st.columns(4)
    with sidebar_column3:
        st.number_input(label = "Vx",
                        key = "vx",
                        placeholder = "Input...",
                        )
        st.number_input(label = "My",
                        key = "my",
                        placeholder = "Input...",
                        )

    with sidebar_column4:
        st.number_input(label = "Vy",
                        key = "vy",
                        placeholder = "Input...",
                        )
        st.number_input(label = "T",
                        key = "t",
                        placeholder = "Input...",
                        )

    with sidebar_column5:
        st.number_input(label = "N",
                        key = "n",
                        placeholder = "Input...",
                        )

        st.selectbox(label = "Seismic Loading",
                    key = "seismic_loading",
                    index = None,
                    placeholder = "Select...",
                    options = (True, False)
                    )

    with sidebar_column6:
        st.number_input(label = "Mx",
                        key = "mx",
                        placeholder = "Input...",
                        )
        st.selectbox(label = "Phi Factor Override",
                     key = "phi_factor_override",
                     index = None,
                     placeholder = "Select",
                     options = (True, False))

    st.header("Installation Conditions")

    #DELETE LATER: sample data for options. delete when real data is fixed
    sample_options = ("Sample 1", "Sample 2", "Sample 3", "Sample 4")

    sidebar_column7, sidebar_column8, sidebar_column9 = st.columns(3)

    with sidebar_column7:
        st.number_input(label = "hef",
                        key = "hef",
                        placeholder = "Input...",
                        )
        st.selectbox(label = "Drilling Type",
                     key = "drilling_type",
                     index = None,
                     placeholder = "Select...",
                     options = sample_options)
    
    with sidebar_column8:
        st.number_input(label = "Short Term Temp",
                        key = "short_term_temp",
                        placeholder = "Input...",
                        )
        st.selectbox(label = "Inspection Condition",
                     key = "inspection_condition",
                     index = None,
                     placeholder = "Select...",
                     options = sample_options)
    
    with sidebar_column9:
        st.number_input(label = "Long Term Temp",
                        key = "long_term_temp",
                        placeholder = "Input...")
        st.selectbox(label = "Moisture Condition",
                     key = "moisture_condition",
                     index = None,
                     placeholder = "Select...",
                     options = sample_options)
    
    st.text_input(label = "Anchor Layout String",
                  key = "anchor_layout_string",
                  placeholder = "Input...")

    if "data_column_counter" not in st.session_state:
        st.session_state.data_column_counter = 0

    if "data_column" not in st.session_state:
        st.session_state.data_column = []

    data = {}
    if st.button("Record Data"):
        st.session_state["data_column_counter"] += 1
        for k, v in st.session_state.items():
            #UPDATE: Keep excluded strings updated with values we don't want to change
            if k not in ("data_column", "data_column_counter"):
                data[k] = v
        st.session_state["data_column"].append(data)
    data_column_df = pd.DataFrame(st.session_state["data_column"])

main_column1, main_column2 = st.columns(2)

with main_column1:
    sample_df = pd.DataFrame({
                            'X': [1, 2, 3],
                            'Y': [4, 5, 6],
                            'Vx': [0.1, 0.2, 0.3],
                            'Vy': [0.4, 0.5, 0.6],
                            'N': [10, 20, 30]
                            })
    sample_df = st.data_editor(sample_df,
                               num_rows = "dynamic",
                               key = "anchor_geometry_and_loads")

with main_column2:
    st.line_chart(sample_df)

st.header("All Project Designs")

st.markdown(r"""
To solve the quadratic equation:

$$
ax^2 + bx + c = 0
$$

We use the quadratic formula:

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
""")

if st.session_state["data_column"] != []:
    project_design_table_order = ["base_material", "deck_installation_location",
                                "hole_diameter_of_fastened_part", "anchor_product_mode",
                                "specified_product", "product_group", "anchor_load_input_location",
                                "vx", "vy", "n", "seismic_loading", "phi_factor_override", "hef",
                                "short_term_temp", "long_term_temp", "drilling_type",
                                "moisture_condition", "anchor_layout_string"]
    temp_df = pd.DataFrame(st.session_state["data_column"])[project_design_table_order]
    temp_df = temp_df.T
    # temp_df = temp_df[project_design_table_order]
    st.dataframe(temp_df, height = 843)

if not sample_df.empty:
    x = sample_df.loc[0, "X"]
    y = sample_df.loc[0, "Y"]
    vx = sample_df.loc[0, "Vx"]
    qf = Quadratic_Formula(x, y, vx)

    quadratic_formula_result_1, quadratic_formula_result_2 = qf.solve()
    st.write(f"Quadratic Equation Results! {quadratic_formula_result_1} {quadratic_formula_result_2}")

if not sample_df.empty:
    x = sample_df.loc[1, "X"]
    y = sample_df.loc[1, "Y"]
    vx = sample_df.loc[1, "Vx"]
    py_th = Pythagorean_Theorem(x, y).solve()

    st.markdown(r"""
In a right-angled triangle, the Pythagorean Theorem states:

$$
a^2 + b^2 = c^2
$$

Where:
- \(a\) and \(b\) are the lengths of the legs,
- \(c\) is the length of the hypotenuse.
""")
    pythagorean_theorem_result = py_th.solve()
    st.write(f"Pythagorean Theorem Results! {pythagorean_theorem_result}")
