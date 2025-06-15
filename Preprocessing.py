import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os

def app():
    def load_data():
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "heart.csv")

        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            return df
        except FileNotFoundError:
            st.error("❌ File 'heart.csv' not found. Please make sure it's uploaded to GitHub in the same folder as app.py")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            st.error("❌ An error occurred while reading the file.")
            st.text(str(e))
            return pd.DataFrame()

    df = load_data()
    if df.empty:
        st.stop()

    df1 = df.copy()

    # Display DataFrame feature Types
    st.subheader("DataFrame feature Types")
    st.dataframe(df.dtypes)

    st.subheader("DataFrame Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Description of Numeric Columns")
    st.dataframe(df.describe().T)

    st.subheader("Description of Object Columns")
    st.dataframe(df.describe(include="object").T)

    st.subheader("Duplicated Rows Count")
    st.text(f"Number of duplicated rows: {df.duplicated().sum()}")

    st.subheader("Missing Values")
    st.dataframe(pd.DataFrame(df.isnull().sum(), columns=['Missing Count']))

    missing_percentage = df1.isnull().sum() / df1.shape[0] * 100
    st.subheader("Percentage of Missing Values per Column")
    st.dataframe(pd.DataFrame(missing_percentage, columns=['Missing Percentage %']))

    with st.expander("Handling Missing Values (Imputation)"):
        st.subheader("Imputing Missing Values in Numeric Columns")
        numeric_col = df1[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].columns
        with st.spinner("⏳ Waiting... Handling missing values..."):
            num_imp = SimpleImputer(strategy='mean')
            df1[numeric_col] = num_imp.fit_transform(df1[numeric_col])
        st.success("Successfully imputed missing values using SimpleImputer (Mean strategy)")
        st.subheader("Missing Values After Imputation")
        st.dataframe(pd.DataFrame(df1.isnull().sum(), columns=['Missing Count']))

    with st.expander("Handling Outliers"):
        st.subheader("Boxplots Before Handling Outliers")
        fig_before, axs_before = plt.subplots(2, 3, figsize=(20, 10))
        axs_before = axs_before.flatten()
        for i, col in enumerate(numeric_col):
            sns.boxplot(x=df1[col], color='skyblue', ax=axs_before[i])
            axs_before[i].set_title(f'Boxplot of {col} (Before)')
        for j in range(i + 1, len(axs_before)):
            fig_before.delaxes(axs_before[j])
        plt.tight_layout()
        st.pyplot(fig_before)

        with st.spinner("⏳ Waiting... Detecting and handling outliers..."):
            Q1 = df1[numeric_col].quantile(0.25)
            Q3 = df1[numeric_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = ((df1[numeric_col] < lower_bound) | (df1[numeric_col] > upper_bound)).any(axis=1)
            outliers = df1[outlier_mask]
            st.subheader("Detected Outliers")
            st.dataframe(outliers)

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
        for j in range(i + 1, len(axs_after)):
            fig_after.delaxes(axs_after[j])
        plt.tight_layout()
        st.pyplot(fig_after)

    df1 = df1[~df1["Sex"].isin(['X', 'Unknown'])]

    with st.spinner("⏳ Waiting... Encoding Categorical Features ..."):
        with st.expander("Encoding Categorical Features (Label Encoding)"):
            categorical_label = df1[['Sex', 'ExerciseAngina']].columns
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
                index=df1.index
            )
            df1 = pd.concat([df1.drop(categorical_OneHot, axis=1), OneHot_encoded_df], axis=1)
            st.write("Categorical features encoded using One-Hot Encoding.")
            st.dataframe(df1.head())

        st.success("✅ Encoding Categorical Features using Label Encoding and One-Hot Encoding.")

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
    st.session_state['processed_df'] = df1.copy()
    st.sidebar.success("✅ Data preprocessing completed.")
