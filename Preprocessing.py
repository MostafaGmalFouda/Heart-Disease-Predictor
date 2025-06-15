import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import io

def app():
    # --- Load Data ---
    path = r"D:\Gemy Study\Programming\Projects\Faculty\Machine learning\Project\heart.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File not found at: {path}. Please check the file path.")
        return
    df1 = df.copy()

    #Display DataFrame feature Types
    st.subheader("DataFrame feature Types")
    st.dataframe(df.dtypes)

    st.subheader("DataFrame Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    #Display DataFrame description (numeric columns)
    st.subheader("Description of Numeric Columns")
    st.dataframe(df.describe().T)

    # Display DataFrame description (object columns)
    st.subheader("Description of Object Columns")
    st.dataframe(df.describe(include="object").T)

    # Display duplicated rows count
    st.subheader("Duplicated Rows Count")
    st.text(f"Number of duplicated rows: {df.duplicated().sum()}")

    # Display missing values
    st.subheader("Missing Values")
    st.dataframe(pd.DataFrame(df.isnull().sum(), columns=['Missing Count']))

    # display missing values percentage
    missing_percentage = df1.isnull().sum()/df1.shape[0]*100
    st.subheader("Percentage of Missing Values per Column")
    st.dataframe(pd.DataFrame(missing_percentage, columns=['Missing Percentage %']))

    with st.expander("Handling Missing Values (Imputation)"):
        st.subheader("Imputing Missing Values in Numeric Columns")

        # Select numeric columns
        numeric_col = df1[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].columns

        with st.spinner("⏳ Waiting... Handling missing values..."):
        # Apply SimpleImputer with mean strategy
            num_imp = SimpleImputer(strategy='mean')
            df1[numeric_col] = num_imp.fit_transform(df1[numeric_col])

        st.success("Successfully imputed missing values using SimpleImputer (Mean strategy)")

        # Show missing values after imputation
        st.subheader("Missing Values After Imputation")
        missing_after = pd.DataFrame(df1.isnull().sum(), columns=['Missing Count'])
        st.dataframe(missing_after)

    with st.expander("Handling Outliers"):

        st.subheader("Boxplots Before Handling Outliers")
        fig_before, axs_before = plt.subplots(2, 3, figsize=(20, 10))
        axs_before = axs_before.flatten()
        for i, col in enumerate(numeric_col):
            sns.boxplot(x=df1[col], color='skyblue', ax=axs_before[i])
            axs_before[i].set_title(f'Boxplot of {col} (Before)')
        for j in range(i+1, len(axs_before)):
            fig_before.delaxes(axs_before[j])
        plt.tight_layout()
        plt.suptitle('Boxplots of Numerical Features Before Outlier Handling', fontsize=18, y=1.02)
        st.pyplot(fig_before)

        
        with st.spinner("⏳ Waiting... Detecting and handling outliers..."):
        # Calculate IQR
            Q1 = df1[numeric_col].quantile(0.25)
            Q3 = df1[numeric_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Detect outliers
            outlier_mask = ((df1[numeric_col] < lower_bound) | (df1[numeric_col] > upper_bound)).any(axis=1)
            outliers = df1[outlier_mask]

            # Show detected outliers
            st.subheader("Detected Outliers")
            st.dataframe(outliers)

            # Replace outliers with median
            for col in numeric_col:
                median = df1[col].median()
                df1.loc[(df1[col] < lower_bound[col]) | (df1[col] > upper_bound[col]), col] = median

        st.write("✅ Outliers in numerical columns have been replaced with the median.")

        
        st.subheader("Boxplots After Handling Outliers")
        fig_after, axs_after = plt.subplots(2, 3, figsize=(20, 10))
        axs_after = axs_after.flatten()
        for i, col in enumerate(numeric_col):
            sns.boxplot(x=df1[col], color='lightcoral', ax=axs_after[i])
            axs_after[i].set_title(f'Boxplot of {col} (After)')
        for j in range(i+1, len(axs_after)):
            fig_after.delaxes(axs_after[j])
        plt.tight_layout()
        plt.suptitle('Boxplots of Numerical Features After Outlier Handling', fontsize=18, y=1.02)
        st.pyplot(fig_after)

    # drop values that are wrong 
    df1 = df1[~df1["Sex"].isin(['X', 'Unknown'])]    

    with st.spinner("⏳ Waiting... Encoding Categorical Features ..."):
        with st.expander("Encoding Categorical Features (Label Encoding)"):
            categorical_label = df1[['Sex','ExerciseAngina']].columns
            label_encoder = LabelEncoder()
            for col in categorical_label:
                df1[col] = label_encoder.fit_transform(df1[col])
            st.write("Categorical features encoded using Label Encoding.")
            st.dataframe(df1.head())
        
        with st.expander("Encoding Categorical Features (One-Hot Encoding)"):
            categorical_OneHot = df1[['ChestPainType', 'RestingECG', 'ST_Slope']].columns
            OneHot_encoder = OneHotEncoder(sparse_output=False)
            OneHot_encoded = OneHot_encoder.fit_transform(df1[categorical_OneHot])
            OneHot_encoded_df = pd.DataFrame(
                OneHot_encoded,
                columns=OneHot_encoder.get_feature_names_out(categorical_OneHot),
                index=df1.index             )
            df1 = pd.concat([df1.drop(categorical_OneHot, axis=1), OneHot_encoded_df], axis=1)
            st.write("Categorical features encoded using One-Hot Encoding.")
            st.dataframe(df1.head())
         
            st.success("✅ Encoding Categorical Features by (Label Encoding) and (One-Hot Encoding).")

    with st.expander("Correlation Heatmap"):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df1.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap")
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)

    with st.expander("Feature Scaling"):
        numeric_col_scaled = df1[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].columns
        scaler = StandardScaler()
        df1[numeric_col_scaled] = scaler.fit_transform(df1[numeric_col_scaled])
        st.write("Numerical features scaled using StandardScaler.")
        st.dataframe(df1.head())
        
    st.session_state['scaler'] = scaler
    st.session_state['processed_df'] =df1.copy()
    st.sidebar.success("Data preprocessing completed.")

