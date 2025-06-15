import streamlit as st
from PIL import Image

def app():
    
    st.title("Heart Disease Prediction Using Machine Learning")
    st.markdown(
        """
        This project aims to develop a machine learning system to predict the presence of heart disease
        using clinical features such as age, chest pain type, cholesterol level, and maximum heart rate.
        By applying preprocessing techniques and training multiple models—including Neural Network, Logistic Regression,
        Random Forest, and XGBoost—the system evaluates and compares their predictive performance. 
        The goal is to assist in early diagnosis and enhance decision-making in cardiovascular healthcare.
        
        """
    )

    # --- Image Frame ---
    try:
        image = Image.open(r"https://github.com/MostafaGmalFouda/Heart-Disease-Predictor/blob/main/Heart.webp")  
        st.image(image, caption="Heart Disease by AI", use_container_width=True)
    except FileNotFoundError:
        st.warning("Image 'Heart.webp' not found. Please place the image in the same directory or update the path.")

    st.sidebar.success("Navigate to different sections using the sidebar.")
