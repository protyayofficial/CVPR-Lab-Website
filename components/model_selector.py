import streamlit as st
from config import AVAILABLE_MODELS

def model_selector():
    """Component for selecting multiple ML models."""
    st.subheader("Select Models")
    
    # Create checkboxes for each model
    selected_models = {}
    for model_name in AVAILABLE_MODELS.keys():
        selected_models[model_name] = st.checkbox(f"Apply {model_name}")
    
    # Get list of selected model names
    selected_model_names = [model_name for model_name, selected in selected_models.items() if selected]
    
    # Display selected models
    if selected_model_names:
        st.write(f"Selected models: {', '.join(selected_model_names)}")
    else:
        st.info("Please select at least one model to process the image")
    
    # Process button - enabled only if at least one model is selected
    process_button = st.button("Process Image", disabled=len(selected_model_names) == 0)
    
    return selected_model_names, process_button