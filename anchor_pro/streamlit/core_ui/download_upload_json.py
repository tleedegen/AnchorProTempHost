import streamlit as st
from anchor_pro.streamlit.core_functions.parameters import initialize_parameter_group
from anchor_pro.streamlit.core_functions.json_serialization import to_json_str, from_json_str

def render_download_upload_ui():
    """
    Renders the Download/Upload section in the sidebar.
    """
    # Ensure session state is initialized
    initialize_parameter_group()

    # Initialize a state to track the processed file if it doesn't exist
    if 'last_loaded_file_signature' not in st.session_state:
        st.session_state['last_loaded_file_signature'] = None

    st.markdown("---")
    st.header("Project File")

    # --- DOWNLOAD (Save) ---
    current_group = st.session_state['parameter_group']
    
    try:
        json_string = to_json_str(current_group)
        st.download_button(
            label="Download Project JSON",
            file_name="anchor_design_project.json",
            mime="application/json",
            data=json_string,
            help="Save your current parameters to a JSON file."
        )
    except Exception as e:
        st.error(f"Failed to prepare download: {e}")

    # --- UPLOAD (Load) ---
    uploaded_file = st.file_uploader("Upload Project JSON", type=["json"])

    if uploaded_file is not None:
        # Create a unique signature for the file (name + size)
        file_signature = (uploaded_file.name, uploaded_file.size)

        # ONLY load if this specific file hasn't been loaded yet
        if st.session_state['last_loaded_file_signature'] != file_signature:
            try:
                # Read and decode
                json_content = uploaded_file.getvalue().decode("utf-8")
                
                # Deserialize
                new_group = from_json_str(json_content)
                
                # Update Session State
                st.session_state['parameter_group'] = new_group
                
                # Update the tracker so we don't reload this file on the next rerun
                st.session_state['last_loaded_file_signature'] = file_signature
                
                # Force a rerun to refresh the UI with new data immediately (optional but recommended)
                st.rerun()
                
            except ValueError as e:
                st.error(f"Error parsing JSON: {e}")
            except Exception as e:
                st.error(f"Unexpected error loading file: {e}")
        
        else:
            # If file is present but already loaded, just show the status
            count = len(st.session_state['parameter_group'].parameters)
            st.info(f"File loaded. Project contains {count} design(s).")
    
    else:
        # Reset the signature if the user removes the file, so they can re-upload it if needed
        st.session_state['last_loaded_file_signature'] = None