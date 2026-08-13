import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# App Configuration
st.set_page_config(page_title="Heart Disease Risk Classifier", layout="wide")
st.title("Heart Disease Risk Classification Application")
st.markdown("""
Upload your **test_data.csv** to evaluate various Machine Learning models. 
The system predicts the risk of Heart Disease (0 = No Risk, 1 = Risk).
""")

# Sidebar settings
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("2. Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a Classification Model:",
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "SVM")
)

# Mapping Dropdown selection to file names
model_file_mapping = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "SVM": "SVM.pkl"
}

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Separate Features and Target
    if 'HeartDiseaseorAttack' not in df.columns:
        st.error("Error: The uploaded CSV must contain the target column 'HeartDiseaseorAttack'.")
    else:
        X_test = df.drop('HeartDiseaseorAttack', axis=1)
        y_test = df['HeartDiseaseorAttack']

        try:
            # Load Scaler & Model
            scaler = joblib.load('model/scaler.pkl')
            model = joblib.load(os.path.join('model', model_file_mapping[model_choice]))

            # Preprocess test data
            X_test_scaled = scaler.transform(X_test)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model.decision_function(X_test_scaled)  # Fallback

            # Compute Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Display Metrics
            st.markdown(f"### Evaluation Metrics for **{model_choice}**")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix Plot
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading models. Ensure 'model/' folder exists with trained .pkl files. Details: {e}")
else:
    st.info("Please upload the 'test_data.csv' file in the sidebar to begin evaluation.")
