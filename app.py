import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("Adult Income Classification")
st.write("Machine Learning Assignment 2 – BITS M.Tech AIML")

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    return models

models = load_models()

if len(models) != 5:
    st.error("Some model files are missing. Check your model/ directory.")
    st.write("Expected model files:")
    st.write(list(MODEL_FILES.values()))
    st.stop()

st.success("All five trained models loaded successfully.")

uploaded_file = st.file_uploader(
    "Upload test CSV file (including income target column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names and text values
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("?", np.nan)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "income" not in df.columns:
        st.error("The CSV must contain the target column 'income'.")
        st.stop()

    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

    valid_labels = {"<=50K", ">50K"}

    if not set(df["income"].dropna().unique()).issubset(valid_labels):
        st.error("Income values must be <=50K or >50K.")
        st.stop()

    X_test = df.drop(columns=["income"])
    y_test = df["income"]

    st.write("Number of test records:", len(df))

    selected_model = st.selectbox(
        "Select a trained model",
        list(models.keys())
    )

    if st.button("Evaluate Model"):
        model = models[selected_model]

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = None

        auc = (
            roc_auc_score((y_test == ">50K").astype(int), y_score)
            if y_score is not None
            else np.nan
        )

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, pos_label=">50K", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, pos_label=">50K", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, pos_label=">50K", zero_division=0
        )
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["<=50K", ">50K"]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual <=50K", "Actual >50K"],
            columns=["Predicted <=50K", "Predicted >50K"]
        )

        st.dataframe(cm_df)

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(pd.DataFrame(report).transpose())

    if st.button("Compare All Models"):
        results = []

        for name, model in models.items():
            predictions = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.decision_function(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(
                    (y_test == ">50K").astype(int),scores
                ),
                "Precision": precision_score(
                    y_test, predictions,
                    pos_label=">50K", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions,
                    pos_label=">50K", zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions,
                    pos_label=">50K", zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, predictions)
            })

        st.subheader("Model Comparison")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
