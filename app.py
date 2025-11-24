import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from joblib import load as joblib_load

# --------------------------
# Paths to your pkl files
# --------------------------
MODEL_PATH = "HR_Attrition_Model_BalancedRF_20251023.joblib"
FEATURES_PATH = "HR_Model_Features_20251023.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. "
                 f"Make sure it is in the same folder as this app.")
        return None
    try:
        model = joblib_load(MODEL_PATH)
    except Exception as e:
        st.error("Error loading model file.")
        st.exception(e)
        return None
    return model

@st.cache_resource
def load_feature_list():
    """Load the feature names used by the model."""
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file not found at {FEATURES_PATH}. "
                 f"Make sure it is in the same folder as this app.")
        return None
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    # If this is a dict or something else, adapt here:
    if isinstance(features, dict) and "features" in features:
        features = features["features"]
    return list(features)

def build_input_ui(feature_names):
    """
    Create Streamlit widgets for feature input or allow pasting a record.
    Returns a single-row DataFrame in the correct column order.
    """
    st.subheader("Employee Information")

    # Let user choose input method
    input_method = st.radio("Select input method:", ["Manual input", "Paste record (CSV/TSV/Dict)"])

    if input_method == "Manual input":
        input_data = {}
        for feat in feature_names:
            # numeric input by default; you can adapt types here
            input_data[feat] = st.number_input(feat, value=0.0)
        row_df = pd.DataFrame([input_data], columns=feature_names)

    else:  # Paste record
        record_text = st.text_area(
            "Paste your record here (CSV row, Tab-separated row, or Python dict/JSON format):",
            height=150
        )
        row_df = pd.DataFrame(columns=feature_names)  # default empty

        if record_text:
            parsed = False
            import io, ast
            # 1️⃣ Try parsing as dict/JSON
            try:
                data_dict = ast.literal_eval(record_text)
                if isinstance(data_dict, dict):
                    row_df = pd.DataFrame([data_dict], columns=feature_names)
                    parsed = True
            except Exception:
                pass

            # 2️⃣ Try parsing as CSV (comma-separated)
            if not parsed:
                try:
                    row_df = pd.read_csv(io.StringIO(record_text), names=feature_names)
                    parsed = True
                except Exception:
                    pass

            # 3️⃣ Try parsing as TSV (tab-separated)
            if not parsed:
                try:
                    row_df = pd.read_csv(io.StringIO(record_text), names=feature_names, sep="\t")
                    parsed = True
                except Exception:
                    pass

            if not parsed:
                st.error("Failed to parse the record. Make sure it's a valid CSV, tab-separated, or dictionary format.")

        # Show preview
        if not row_df.empty:
            with st.expander("Preview input DataFrame"):
                st.dataframe(row_df)

    return row_df



def main():
    st.set_page_config(
        page_title="HR Attrition Prediction",
        page_icon="📊",
        layout="centered"
    )

    st.title("📊 HR Attrition Prediction App")
    st.write(
        """
        This app uses a trained machine learning model to predict the **likelihood of employee attrition**.
        
        1. Fill in the employee details below  
        2. Click **Predict Attrition**  
        3. View the predicted probability and label
        """
    )

    model = load_model()
    feature_names = load_feature_list()

    if model is None or feature_names is None:
        st.stop()

    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.info(
        """
        - Model: Balanced Random Forest  
        - Files:
            - `HR_Attrition_Model_BalancedRF_20251023.pkl`  
            - `HR_Model_Features_20251023.pkl`
        """
    )

    # Build the UI for feature input
    with st.form("attrition_form"):
        input_df = build_input_ui(feature_names)

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        with st.spinner("Running prediction..."):
            try:
                # If your model has a preprocessing pipeline inside, this
                # should work directly. Otherwise, apply your preprocessing here.
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]
                    # Assuming binary [No, Yes] order. Adjust index if reversed.
                    attrition_prob = float(proba[1])
                else:
                    # If no predict_proba, use decision_function or predict
                    pred_raw = model.predict(input_df)[0]
                    # Fake probability from decision_function or treat as label
                    attrition_prob = float(pred_raw)

                pred_label = "Yes" if attrition_prob >= 0.5 else "No"

                st.subheader("Prediction Result")
                st.metric("Attrition Predicted", pred_label)
                st.progress(int(attrition_prob * 100))
                st.write(f"Estimated probability of attrition: **{attrition_prob:.2%}**")

                with st.expander("Show raw input and model-ready data"):
                    st.write("Input DataFrame sent to model:")
                    st.dataframe(input_df)

            except Exception as e:
                st.error("An error occurred during prediction.")
                st.exception(e)

    st.markdown("---")
    st.caption("Built with Streamlit · HR Attrition Demo")

if __name__ == "__main__":
    main()
