import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------
# Paths to your pkl files
# --------------------------
MODEL_PATH = "HR_Attrition_Model_BalancedRF_20251023.pkl"
FEATURES_PATH = "HR_Model_Features_20251023.pkl"

@st.cache_resource
def load_model():
    """Load the trained model from pickle."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. "
                 f"Make sure it is in the same folder as this app.")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
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
    Create Streamlit widgets for all features and return
    a single-row DataFrame in the correct column order.
    
    🔴 IMPORTANT:
    Update this function so the widget types & default values
    match your actual feature set and preprocessing.
    """
    st.subheader("Employee Information")

    # Container to collect raw values
    input_data = {}

    # Example mapping: adjust based on your project
    # You can add if/elif by feature name or use patterns.
    for feat in feature_names:
        # Simple heuristic to decide widget type
        lower = feat.lower()

        # Example for common HR fields – tweak/remove as needed
        if "age" in lower:
            input_data[feat] = st.number_input(feat, min_value=18, max_value=70, value=30)
        elif "income" in lower or "salary" in lower:
            input_data[feat] = st.number_input(feat, min_value=0, max_value=1_000_000, value=5000)
        elif "distance" in lower:
            input_data[feat] = st.number_input(feat, min_value=0, max_value=100, value=10)
        elif "years" in lower or "tenure" in lower:
            input_data[feat] = st.number_input(feat, min_value=0, max_value=40, value=3)
        elif "overtime" in lower:
            input_data[feat] = st.selectbox(feat, ["No", "Yes"])
        elif "gender" in lower:
            input_data[feat] = st.selectbox(feat, ["Male", "Female"])
        elif "marital" in lower:
            input_data[feat] = st.selectbox(feat, ["Single", "Married", "Divorced"])
        elif "education" in lower:
            input_data[feat] = st.selectbox(feat, ["Below College", "College", "Bachelor", "Master", "Doctor"])
        elif "travel" in lower:
            input_data[feat] = st.selectbox(feat, ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        elif "jobrole" in lower or "role" in lower:
            input_data[feat] = st.text_input(feat, "Sales Executive")
        elif "department" in lower:
            input_data[feat] = st.selectbox(feat, ["Sales", "Research & Development", "Human Resources"])
        elif "satisfaction" in lower or "rating" in lower or "score" in lower:
            input_data[feat] = st.slider(feat, min_value=1, max_value=5, value=3)
        else:
            # Fallback: numeric input, you can change this to text_input
            input_data[feat] = st.text_input(feat, "")

    # Convert dictionary to DataFrame with a single row
    row_df = pd.DataFrame([input_data], columns=feature_names)
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
