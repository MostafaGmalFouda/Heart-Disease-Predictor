import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io 
import os
def app():
  def load_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "heart.csv")

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        import streamlit as st
        st.error("❌ File 'heart.csv' not found. Please make sure it's uploaded to GitHub in the same folder as app.py")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        st.error("❌ An error occurred while reading the file.")
        st.text(str(e))
        return pd.DataFrame()

   
    st.subheader("Original Data Overview")
    num_rows = st.slider("Select number of rows to display:", min_value=5, max_value=len(df), value=10, step=5)
    st.dataframe(df.head(num_rows)) 

    # plot the histograms of numerical features
    st.subheader("Histograms of Numerical Features")
    numeric_col = df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].columns
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()  # Flatten for easy indexing

    for i, col in enumerate(numeric_col):
        sns.histplot(data=df, x=col, bins=20, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f'Histogram of {col}', fontsize=16, fontweight='bold')
        axes[i].set_xlabel(col, fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)
        axes[i].tick_params(axis='both', labelsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Hide the last subplot if unused (since 3x2 = 6 but only 5 plots)
    if len(numeric_col) < len(axes):
        for j in range(len(numeric_col), len(axes)):
            fig.delaxes(axes[j])

    fig.suptitle('Histograms of Numerical Features', fontsize=18, y=1.02)
    fig.tight_layout()
    st.pyplot(fig)

    # plot the countplot of categorical features
    st.subheader("Countplot of Categorical Features")
    df3 = df.copy()
    df3 = df3[~df3["Sex"].isin(['X', 'Unknown'])]
    categorical_col = df3.select_dtypes(include=['object']).columns

    sns.set_theme(style="darkgrid")
    fig_0, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, col in enumerate(categorical_col):
        sns.countplot(x=col, data=df3, hue="HeartDisease", palette='pastel', ax=axes[i])
        axes[i].set_title(f'Count plot of {col}', fontsize=14, fontweight='bold')
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].tick_params(axis='both', labelsize=12)

    # Hide any unused subplot
    if len(categorical_col) < len(axes):
        for j in range(len(categorical_col), len(axes)):
            fig_0.delaxes(axes[j])
    fig_0.suptitle('Count Plots of Categorical Features', fontsize=18, y=1.02)
    fig_0.tight_layout()
    st.pyplot(fig_0)


    # plot the Heatmap of correlation between numeric columns
    st.subheader("Heatmap of correlation between numeric columns")
    fig_1, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, ax=ax)
    st.pyplot(fig_1)


    st.subheader("Choose the type of visualization you want to perform")
    numiric_cols = df[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']].columns.tolist()
    tab1, tab2, tab3, tab4 = st.tabs(["scatter plot", "histogram", "box plot", "line plot"])
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("choose x axis", numiric_cols)
        with col2:
            y = st.selectbox("choose y axis", numiric_cols)
        with col3:
            color_user = st.selectbox("choose color", df.columns.tolist()) 
        fig_2 = px.scatter(df, x=x, y=y,color = color_user  ,title=f"Scatter plot of {x} vs {y}")
        st.plotly_chart(fig_2)
    with tab2:    
        input_hist = st.selectbox("choose hist value", numiric_cols)
        fig_3 = px.histogram(df, x=input_hist, title=f"Histogram of {input_hist}")
        st.plotly_chart(fig_3)
    with tab3:
        input_box = st.selectbox("choose box value", numiric_cols)
        fig_4 = px.box(df, y=input_box, title=f"Box plot of {input_box}")
        st.plotly_chart(fig_4)
    with tab4:
        col4, col5 = st.columns(2)
        with col4:
            x_line = st.selectbox("choose x line value", numiric_cols)
        with col5:
            y_line = st.selectbox("choose y line value", numiric_cols)
        fig_5 = px.line(df, x=x_line, y=y_line, title=f"Line plot of {x_line} vs {y_line}")
        st.plotly_chart(fig_5)
        
    categorical = df3.select_dtypes(include='object').columns
    tab5, tab6 = st.tabs(["count plot", "pie chart"])
    with tab5:
        col6, col7 = st.columns(2)
        with col6:
            input_count = st.selectbox("choose count value", categorical)
        with col7:
            input_count_hue = st.selectbox("choose hue value", categorical)
        fig_6 = px.histogram(df3, x=input_count, color=input_count_hue, title=f"Count plot of {input_count} by {input_count_hue}")
        st.plotly_chart(fig_6)
    with tab6:
        input_pie = st.selectbox("choose pie value", categorical)
        fig_7 = px.pie(df3, names=input_pie, title=f"Pie chart of {input_pie}")
        st.plotly_chart(fig_7)
    

    st.sidebar.success("Explore the data visually.")
