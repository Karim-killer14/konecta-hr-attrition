# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HR Attrition Training Dashboard", layout="wide")

st.title("🎯 HR Attrition Model - Full Training Dashboard")
st.markdown("This app runs the complete Week 3 training pipeline and displays all results interactively.")

# ========================
# SIDEBAR CONFIGURATION
# ========================
st.sidebar.header("Configuration")
REPLACEMENT_COST = st.sidebar.number_input("Replacement Cost (EGP)", value=90000)
INTERVENTION_COST = st.sidebar.number_input("Intervention Cost (EGP)", value=5000)
RETENTION_SUCCESS_RATE = st.sidebar.slider("Retention Success Rate", 0.0, 1.0, 0.5)

uploaded_file = st.sidebar.file_uploader("Upload Final Payroll CSV", type=["csv"])

run_button = st.sidebar.button("🚀 Run Full Training Pipeline")

# ========================
# MAIN PIPELINE
# ========================
if run_button and uploaded_file:
    st.info("Running full training pipeline... This may take a few minutes.")

    df = pd.read_csv(uploaded_file)

    drop_cols = [
        'Employee_ID', 'National_ID', 'Insurance_Number', 'Email', 'Mobile',
        'Starting_Date', 'Last_Working_Date', 'Attrition_Probability',
        'Job_Title', 'Department', 'Tenure_Category', 'Age_Category', 
        'Salary_Category', 'Education_Level', 'Marital_Status', 'Gender',
        'Working_Conditions'
    ]

    target = "Attrition"
    X = df.drop(columns=[target] + drop_cols, errors='ignore')
    y = df[target]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X.columns = X.columns.str.replace('[\\[\\]<>() ]', '_', regex=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    samplers = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(random_state=42, n_neighbors=3),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3)
    }

    quick_model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss', verbosity=0)
    best_sampler_name = max(samplers, key=lambda s: roc_auc_score(y_val, quick_model.fit(*samplers[s].fit_resample(X_train, y_train)).predict_proba(X_val)[:,1]))
    best_sampler = samplers[best_sampler_name]

    X_train_balanced, y_train_balanced = best_sampler.fit_resample(X_train, y_train)

    models = {
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
        'BalancedRF': BalancedRandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_balanced if name != 'BalancedRF' else X_train, y_train_balanced if name != 'BalancedRF' else y_train)
        prob = model.predict_proba(X_val)[:,1]
        results[name] = roc_auc_score(y_val, prob)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    y_val_prob = best_model.predict_proba(X_val)[:,1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    roi_scores = []
    for t in thresholds:
        y_pred = (y_val_prob >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tp = cm[1,1] if cm.shape[0]>1 else 0
        fp = cm[0,1] if cm.shape[0]>1 else 0
        total_leavers = y_val.sum()
        cost_baseline = total_leavers * REPLACEMENT_COST
        cost_interventions = (tp + fp) * INTERVENTION_COST
        employees_saved = int(tp * RETENTION_SUCCESS_RATE)
        cost_replacements = (total_leavers - employees_saved) * REPLACEMENT_COST
        cost_with_model = cost_interventions + cost_replacements
        roi = ((cost_baseline - cost_with_model) / cost_with_model) if cost_with_model > 0 else 0
        roi_scores.append(roi)

    optimal_threshold = thresholds[np.argmax(roi_scores)]

    y_test_prob = best_model.predict_proba(X_test)[:,1]
    y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

    st.subheader("📊 Final Model Results")
    st.write(f"Best Model: {best_model_name}")
    st.write(f"Optimal Threshold: {optimal_threshold:.2f}")
    st.write(f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.3f}")
    st.write(f"Recall: {recall_score(y_test, y_test_pred):.2%}")

    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"HR_Attrition_Model_{timestamp}.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(f"HR_Model_Features_{timestamp}.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    st.success("✅ Training and artifacts saved successfully")

else:
    st.warning("Upload dataset and click Run to start training")
