# streamlit_app_cleaned.py
"""
Production-ready Streamlit app for HR Attrition Model training pipeline.
Features:
- Clear structure (functions for load/preprocess/train/eval/save)
- Robust error handling
- Caching for data loading/preprocessing
- Safe handling of samplers and models
- ROI-based threshold selection
- Saves artifacts (model, features, config, plots) and offers zip download

Dependencies (pip):
streamlit, pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, matplotlib, seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="HR Attrition Training Dashboard", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    """Load CSV into dataframe (cached)."""
    return pd.read_csv(uploaded_file)


def sanitize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.replace('[\[\]<>() ]', '_', regex=True)
    return df


def preprocess(df, target, drop_cols):
    df = df.copy()
    X = df.drop(columns=[target] + drop_cols, errors='ignore')
    y = df[target]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X = sanitize_columns(X)
    return X, y


def select_best_sampler(X_train, y_train, X_val, y_val, samplers):
    """Quick comparison using a lightweight classifier to pick best sampler by ROC-AUC on validation."""
    scores = {}
    quick_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)

    for name, sampler in samplers.items():
        try:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            quick_model.fit(X_res, y_res)
            if hasattr(quick_model, 'predict_proba'):
                prob = quick_model.predict_proba(X_val)[:, 1]
                scores[name] = roc_auc_score(y_val, prob)
            else:
                # fallback to predict
                prob = quick_model.predict(X_val)
                scores[name] = roc_auc_score(y_val, prob)
        except Exception:
            scores[name] = -1

    # choose best; if none succeeded return None
    best = max(scores, key=lambda k: scores[k]) if len(scores) > 0 and max(scores.values()) > 0 else None
    return best


def train_and_evaluate(X_train_bal, y_train_bal, X_train, y_train, X_val, y_val):
    models = {
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        'BalancedRF': BalancedRandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    }

    results = {}
    for name, model in models.items():
        try:
            if name == 'BalancedRF':
                model.fit(X_train, y_train)
            else:
                model.fit(X_train_bal, y_train_bal)

            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_val)[:, 1]
            else:
                prob = model.predict(X_val)

            results[name] = roc_auc_score(y_val, prob)
        except Exception as e:
            results[name] = -1

    best_name = max(results, key=results.get)
    best_model = models[best_name]
    return best_name, best_model, results


def compute_optimal_threshold(y_val, y_val_prob, replacement_cost, intervention_cost, retention_success_rate):
    thresholds = np.arange(0.1, 0.9, 0.01)
    roi_scores = []
    for t in thresholds:
        y_pred = (y_val_prob >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fp = cm[0, 1] if cm.shape[0] > 1 else 0
        total_leavers = int(y_val.sum())
        cost_baseline = total_leavers * replacement_cost
        cost_interventions = (tp + fp) * intervention_cost
        employees_saved = int(tp * retention_success_rate)
        cost_replacements = (total_leavers - employees_saved) * replacement_cost
        cost_with_model = cost_interventions + cost_replacements
        roi = ((cost_baseline - cost_with_model) / cost_with_model) if cost_with_model > 0 else 0
        roi_scores.append(roi)

    opt_idx = int(np.nanargmax(roi_scores))
    return thresholds[opt_idx], thresholds, roi_scores


def save_artifacts(output_dir, timestamp, model, feature_list, config):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"HR_Attrition_Model_{timestamp}.pkl")
    features_path = os.path.join(output_dir, f"HR_Model_Features_{timestamp}.pkl")
    config_path = os.path.join(output_dir, f"Training_Config_{timestamp}.txt")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(features_path, 'wb') as f:
        pickle.dump(feature_list, f)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    return model_path, features_path, config_path


def make_zip(files, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in files:
            if os.path.exists(f):
                zipf.write(f, arcname=os.path.basename(f))
    return zip_path


# -------------------------
# UI
# -------------------------
st.title("🎯 HR Attrition Model - Full Training Dashboard (Cleaned)")
st.markdown("This app runs a full training pipeline and lets you download artifacts.")

st.sidebar.header("Configuration")
REPLACEMENT_COST = st.sidebar.number_input("Replacement Cost (EGP)", value=90000)
INTERVENTION_COST = st.sidebar.number_input("Intervention Cost (EGP)", value=5000)
RETENTION_SUCCESS_RATE = st.sidebar.slider("Retention Success Rate", 0.0, 1.0, 0.5)

uploaded_file = st.sidebar.file_uploader("Upload Final Payroll CSV", type=["csv"] )
run_button = st.sidebar.button("🚀 Run Full Training Pipeline")

OUTPUT_DIR = "outputs"

if run_button:
    if uploaded_file is None:
        st.warning("Please upload a CSV file before running the pipeline.")
    else:
        try:
            with st.spinner("Loading data..."):
                df = load_csv(uploaded_file)

            # columns to drop (keeps errors='ignore')
            drop_cols = [
                'Employee_ID', 'National_ID', 'Insurance_Number', 'Email', 'Mobile',
                'Starting_Date', 'Last_Working_Date', 'Attrition_Probability',
                'Job_Title', 'Department', 'Tenure_Category', 'Age_Category', 
                'Salary_Category', 'Education_Level', 'Marital_Status', 'Gender',
                'Working_Conditions'
            ]

            target = 'Attrition'
            if target not in df.columns:
                st.error(f"Target column '{target}' not found in uploaded file.")
            else:
                with st.spinner("Preprocessing..."):
                    X, y = preprocess(df, target, drop_cols)

                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

                samplers = {
                    'SMOTE': SMOTE(random_state=42, k_neighbors=3),
                    'ADASYN': ADASYN(random_state=42, n_neighbors=3),
                    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3)
                }

                with st.spinner("Selecting best sampler..."):
                    best_sampler_name = select_best_sampler(X_train, y_train, X_val, y_val, samplers)

                if best_sampler_name is None:
                    st.warning("No sampler improved validation AUC. Proceeding without resampling.")
                    X_train_bal, y_train_bal = X_train, y_train
                    best_sampler_name = 'None'
                else:
                    best_sampler = samplers[best_sampler_name]
                    X_train_bal, y_train_bal = best_sampler.fit_resample(X_train, y_train)

                st.info(f"Best Sampler: {best_sampler_name}")

                with st.spinner("Training models and selecting best..."):
                    best_model_name, best_model, model_scores = train_and_evaluate(X_train_bal, y_train_bal, X_train, y_train, X_val, y_val)

                st.write("### Model scores (validation ROC-AUC)")
                st.table(pd.DataFrame.from_dict(model_scores, orient='index', columns=['roc_auc']).sort_values('roc_auc', ascending=False))

                st.success(f"Best model on validation: {best_model_name}")

                # fit best model on appropriate training data (if needed it's already fitted in train_and_evaluate)
                # but to be safe, retrain on full training set (balanced if applicable)
                try:
                    if best_model_name == 'BalancedRF':
                        best_model.fit(X_train, y_train)
                    else:
                        best_model.fit(X_train_bal, y_train_bal)
                except Exception:
                    pass

                # predict probabilities on validation to choose threshold
                if hasattr(best_model, 'predict_proba'):
                    y_val_prob = best_model.predict_proba(X_val)[:, 1]
                else:
                    y_val_prob = best_model.predict(X_val)

                optimal_threshold, thresholds, roi_scores = compute_optimal_threshold(y_val, y_val_prob, REPLACEMENT_COST, INTERVENTION_COST, RETENTION_SUCCESS_RATE)

                st.write(f"**Optimal threshold (ROI-based):** {optimal_threshold:.2f}")

                # Evaluate on test
                if hasattr(best_model, 'predict_proba'):
                    y_test_prob = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_test_prob = best_model.predict(X_test)

                y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

                st.subheader("📊 Final Model Results (Test)")
                try:
                    test_auc = roc_auc_score(y_test, y_test_prob)
                except Exception:
                    test_auc = float('nan')

                st.write(f"Best Model: {best_model_name}")
                st.write(f"Optimal Threshold: {optimal_threshold:.2f}")
                st.write(f"ROC-AUC: {test_auc:.3f}")

                try:
                    st.write(f"Recall: {recall_score(y_test, y_test_pred):.2%}")
                except Exception:
                    st.write("Recall: N/A")

                cm = confusion_matrix(y_test, y_test_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

                # Save artifacts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config = {
                    'replacement_cost': REPLACEMENT_COST,
                    'intervention_cost': INTERVENTION_COST,
                    'retention_success_rate': RETENTION_SUCCESS_RATE,
                    'best_sampler': best_sampler_name,
                    'best_model': best_model_name,
                    'optimal_threshold': float(optimal_threshold)
                }

                model_path, features_path, config_path = save_artifacts(OUTPUT_DIR, timestamp, best_model, X.columns.tolist(), config)

                # save the results plot
                results_plot_path = os.path.join(OUTPUT_DIR, f"HR_Attrition_Model_Week3_Results_{timestamp}.png")
                fig.savefig(results_plot_path, bbox_inches='tight')

                files_to_download = [model_path, features_path, config_path, results_plot_path]

                zip_path = os.path.join(OUTPUT_DIR, f"training_outputs_{timestamp}.zip")
                make_zip(files_to_download, zip_path)

                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download ALL Outputs (ZIP)",
                        data=f,
                        file_name=os.path.basename(zip_path)
                    )

                st.success("✅ Training complete and artifacts prepared.")

        except Exception as e:
            st.exception(e)

else:
    st.info("Upload dataset and click Run to start training")
