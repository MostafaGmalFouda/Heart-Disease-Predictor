import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def evaluate_model(y_true, y_pred, model_name):
    st.subheader(f"{model_name} - Evaluation")
    st.write("Accuracy:", round(accuracy_score(y_true, y_pred), 2))

    st.write("Classification Report:")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Heart Disease', 'Heart Disease'],
                yticklabels=['No Heart Disease', 'Heart Disease'], ax=ax)
    st.pyplot(fig)

def app():
    st.title("🧠 Model Training")

    model_type = st.selectbox('Select Model', ['Neural Network', 'Logistic Regression', 'Random Forest', 'XGBoost'], index=0)
    st.session_state['model_type'] = model_type

    if 'processed_df' not in st.session_state:
        st.warning("Please complete preprocessing first.")
        st.stop()

    df2 = st.session_state['processed_df'].copy()

    X = df2.drop('HeartDisease', axis=1)
    y = df2['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Train Model"):
        if model_type == 'Neural Network':
            st.subheader("Training Neural Network")
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            y_pred_classes = (y_pred > 0.5).astype(int)
            evaluate_model(y_test, y_pred_classes, "Neural Network")
            st.session_state['model'] = model

        elif model_type == 'Logistic Regression':
            st.subheader("Training Logistic Regression")
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred, "Logistic Regression")
            st.session_state['model'] = model

        elif model_type == 'Random Forest':
            st.subheader("Training Random Forest")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred, "Random Forest")
            st.session_state['model'] = model

        elif model_type == 'XGBoost':
            st.subheader("Training XGBoost")
            model = XGBClassifier(eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluate_model(y_test, y_pred, "XGBoost")
            st.session_state['model'] = model

        st.session_state['model_columns'] = X.columns.tolist()

    st.markdown("---")
    st.subheader("💡 Heart Disease Prediction")

    cat_label = ['Sex', 'ExerciseAngina']
    cat_onehot = ['ChestPainType', 'RestingECG', 'ST_Slope']
    num_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

    cat_options = {
        'Sex': ['M', 'F'],
        'ExerciseAngina': ['N', 'Y'],
        'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
        'RestingECG': ['Normal', 'ST', 'LVH'],
        'ST_Slope': ['Up', 'Flat', 'Down']
    }

    original_inputs = {}
    for col in cat_label + cat_onehot:
        original_inputs[col] = st.selectbox(f"Select {col}", cat_options[col])

    for col in num_features:
        if col == 'FastingBS':
            original_inputs[col] = st.selectbox(f"Select {col}", [0, 1])
        else:
            original_inputs[col] = st.number_input(f"Enter {col}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([original_inputs])

        for col in cat_label:
            le = LabelEncoder()
            le.fit(cat_options[col])
            input_df[col] = le.transform([input_df[col][0]])

        encoded_dfs = []
        for col in cat_onehot:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(pd.DataFrame(cat_options[col], columns=[col]))
            transformed = ohe.transform(input_df[[col]])
            encoded_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out([col]), index=input_df.index)
            encoded_dfs.append(encoded_df)

        onehot_df = pd.concat(encoded_dfs, axis=1)
        input_df = pd.concat([input_df.drop(columns=cat_onehot), onehot_df], axis=1)

        if 'model' not in st.session_state or 'scaler' not in st.session_state or 'model_columns' not in st.session_state:
            st.error("❌ Please train the model first after preprocessing.")
            st.stop()

        num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        scaler = st.session_state['scaler']
        input_df[num_features] = scaler.transform(input_df[num_features])

        model_columns = st.session_state['model_columns']
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        model = st.session_state['model']
        trained_model_type = st.session_state.get('model_type', '')

        if trained_model_type == 'Neural Network':
            pred = model.predict(input_df)[0][0]
            label = 'Heart Disease' if pred > 0.5 else 'No Heart Disease'
        else:
            pred = model.predict(input_df)[0]
            label = 'Heart Disease' if pred == 1 else 'No Heart Disease'

        st.success(f"✅ Prediction: {label}")
