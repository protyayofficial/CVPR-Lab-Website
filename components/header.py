import streamlit as st
from config import HEADER_TITLE, HEADER_SUBTITLE
import os
from PIL import Image # <-- Import Pillow

def render_header():
    """Render the application header."""
    col1, col2 = st.columns([1, 3])
    logo_path = "assets/iitrpr_logo.png"

    with col1:
        if os.path.exists(logo_path):
            try:
                original_logo = Image.open(logo_path)
                original_logo = original_logo.convert("RGBA")
                white_bg = Image.new("RGBA", original_logo.size, "WHITE")

                white_bg.paste(original_logo, (0, 0), original_logo)
                st.image(white_bg, width=100)

            except Exception as e:
                st.error(f"Error processing logo: {e}")
                st.markdown(" L ") # Fallback placeholder on error
        else:
            # Corrected the path in the warning message to match the variable used
            st.warning(f"Logo not found at: {logo_path}")
            st.markdown(" L ")

    with col2:
        st.title(HEADER_TITLE)
        st.write(HEADER_SUBTITLE)

    st.divider()

