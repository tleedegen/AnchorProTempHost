import streamlit as st
import tempfile
import os
import shutil
from datetime import datetime
import pandas as pd
from pylatex import Document, Package, NoEscape, Command

from anchor_pro.streamlit.core_functions.calculation_report import CalculationReport
from anchor_pro.streamlit.core_functions.parameters import Parameters
from anchor_pro.streamlit.core_functions.concrete_anchorage_calcs import evaluate_concrete_anchors, create_anchor_props
from anchor_pro.elements.concrete_anchors import ConcreteAnchors, GeoProps, ConcreteProps
from anchor_pro.streamlit.core_ui.sidebar import load_anchor_catalog

def render_report_section():
    """
    Renders the Report Generation section in the Streamlit app.
    References session_state directly for parameters and catalog.
    Generates a combined PDF package for all parameters in the group.
    """
    st.markdown("---")
    st.header("📄 Calculation Package")

    # 1. Access Data from Session State
    param_group = st.session_state.get('parameter_group')
    df_catalog = load_anchor_catalog()

    if not param_group or not param_group.parameters:
        st.info("No parameters defined. Please add at least one calculation to the group.")
        return

    if df_catalog is None:
        st.error("Anchor catalog not found in session state.")
        return

    # 2. Project Information Inputs
    with st.expander("Project Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            project_title = st.text_input("Project Title", value="Anchor Design Calculation")
            job_number = st.text_input("Job Number", value="")
            engineer = st.text_input("Engineer", value="")
        with col2:
            address = st.text_input("Address", value="")
            city = st.text_input("City/State/Zip", value="")
            date_str = st.date_input("Date", value=datetime.today()).strftime('%Y-%m-%d')

        st.write("Additional Notes")
        note1 = st.text_input("Note 1", placeholder="e.g. Building A, Level 2")
        note2 = st.text_input("Note 2", placeholder="e.g. Design Package 1")

    project_info = {
        'project_title': project_title,
        'job_number': job_number,
        'address': address,
        'city': city,
        'project_info1': f"Date: {date_str}",
        'project_info2': f"Engineer: {engineer}" if engineer else "",
        'project_info3': note1,
        'project_info4': note2,
        'package_info1': "Concrete Anchorage Design",
    }

    # 3. Generation Logic
    if st.button("Generate Calculation Package", type="primary"):
        with st.spinner("Generating calculation package..."):
            try:
                # Create a temporary directory for the entire process
                with tempfile.TemporaryDirectory() as tmp_dir:
                    pdf_files = []
                    
                    # Iterate through all parameters in the group
                    progress_bar = st.progress(0)
                    total_params = len(param_group.parameters)

                    for i, param in enumerate(param_group.parameters):
                        # A. Run Calculation
                        # We re-run evaluation to ensure results are fresh and consistent
                        results = evaluate_concrete_anchors(param, df_catalog)

                        # B. Reconstruct Anchor Object (Required for Report/Plotting)
                        # Replicating logic from evaluate_concrete_anchors to get the object
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
                        anchor_obj = ConcreteAnchors(geo_props=geo_props, concrete_props=concrete_props)
                        
                        if param.selected_anchor_id:
                            anchor_row = df_catalog[df_catalog['anchor_id'] == param.selected_anchor_id].iloc[0]
                            mech_props = create_anchor_props(anchor_row, anchor_obj)
                            anchor_obj.set_anchor_props(mech_props)

                        # C. Generate Individual Report
                        report = CalculationReport(
                            project_info=project_info,
                            parameters=param,
                            results=results,
                            anchor_obj=anchor_obj
                        )
                        
                        # Set output to a specific subfolder or unique name
                        report.output_path = tmp_dir
                        safe_name = "".join(x for x in param.name if x.isalnum() or x in " _-").strip()
                        report.file_name = f"{i:02d}_{safe_name}"
                        
                        report.generate_report()
                        
                        pdf_path = os.path.join(tmp_dir, f"{report.file_name}.pdf")
                        if os.path.exists(pdf_path):
                            pdf_files.append(pdf_path)
                        
                        progress_bar.progress((i + 1) / total_params)

                    # D. Merge PDFs into a Binder
                    if pdf_files:
                        merge_doc = Document(geometry_options={"margin": "0in"})
                        merge_doc.packages.append(Package('pdfpages'))
                        
                        for pdf_file in pdf_files:
                            # pypdf/pylatex path handling for includepdf
                            # We use forward slashes for latex compatibility
                            clean_path = pdf_file.replace('\\', '/')
                            merge_doc.append(NoEscape(r'\includepdf[pages=-]{' + clean_path + r'}'))
                        
                        binder_name = "Calculation_Package"
                        binder_path_no_ext = os.path.join(tmp_dir, binder_name)
                        merge_doc.generate_pdf(binder_path_no_ext, clean_tex=False)
                        
                        final_pdf_path = f"{binder_path_no_ext}.pdf"
                        
                        if os.path.exists(final_pdf_path):
                            with open(final_pdf_path, "rb") as f:
                                st.session_state['generated_report_bytes'] = f.read()
                            st.success(f"Package generated with {len(pdf_files)} calculation(s)!")
                        else:
                            st.error("Failed to generate merged PDF.")
                    else:
                        st.warning("No reports were generated.")

            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
                # Optional debugging
                # import traceback
                # st.text(traceback.format_exc())

    # 4. Persistent Download Button
    if 'generated_report_bytes' in st.session_state:
        st.download_button(
            label="Download Package PDF",
            data=st.session_state['generated_report_bytes'],
            file_name="Calculation_Package.pdf",
            mime="application/pdf"
        )