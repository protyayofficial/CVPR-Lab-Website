import streamlit as st
import os
import time
from PIL import Image
from config import AVAILABLE_MODELS, OUTPUT_DIR
from models.inference import process_image

def display_results(input_image, selected_models):
    """Display and manage the results from multiple model processing."""
    if not selected_models:
        st.warning("No models selected. Please select at least one model.")
        return
    
    st.subheader("Processing Results")
    
    # Show processing message
    with st.spinner(f"Processing with {len(selected_models)} model(s)..."):
        # Process the image with each selected model
        results = {}
        
        # Create a progress bar
        progress_bar = st.progress(0)
        progress_step = 1.0 / (len(selected_models) * 100)
        progress_value = 0
        
        # Process each model
        for i, model_name in enumerate(selected_models):
            st.write(f"Processing with {model_name}...")
            
            # Get the model identifier
            model_id = AVAILABLE_MODELS[model_name]
            
            try:
                # Process the image with the current model
                result_image_path = process_image(input_image, model_id)
                results[model_name] = result_image_path
                
                # Simulate processing time with progress updates
                for j in range(100):
                    progress_value += progress_step
                    progress_bar.progress(min(progress_value, 1.0))
                    time.sleep(0.01)
                
            except Exception as e:
                st.error(f"Error processing image with {model_name}: {str(e)}")
        
        # Display results after all processing is complete
        if results:
            st.success(f"✅ Processing complete! Generated {len(results)} results.")
            
            # Display each result in a new section
            for model_name, result_path in results.items():
                with st.expander(f"Result from {model_name}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Original image
                        st.image(input_image, caption="Original Image", use_column_width=True)
                    
                    with col2:
                        # Result image
                        if os.path.exists(result_path):
                            result_image = Image.open(result_path)
                            st.image(result_image, caption=f"{model_name} Result", use_column_width=True)
                            
                            # Add download button for the processed image
                            with open(result_path, "rb") as file:
                                btn = st.download_button(
                                    label=f"Download {model_name} Result",
                                    data=file,
                                    file_name=os.path.basename(result_path),
                                    mime="image/png",
                                    key=f"download_{model_name}"
                                )
                        else:
                            st.error(f"Failed to generate result image for {model_name}.")
        else:
            st.error("Failed to generate any results.")