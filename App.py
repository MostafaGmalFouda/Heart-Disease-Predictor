import streamlit as st

import Home
import Visualization
import Preprocessing
import Modeling

PAGES = {
    "Home": Home,
    "Data Visualization": Visualization,
    "Data Preprocessing": Preprocessing,
    "Model Training & Prediction": Modeling
}

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
for page_name in PAGES.keys():
    if st.sidebar.button(page_name):
        st.session_state['selected_page'] = page_name

# Default to first page if not set
selected_page = st.session_state.get('selected_page', list(PAGES.keys())[0])

# Render the selected page
PAGES[selected_page].app()
