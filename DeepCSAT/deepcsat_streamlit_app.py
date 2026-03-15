import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# DeepCSAT Streamlit App
# Local deployment app for ongoing CSAT prediction and insights
#
# Expected files in the same folder:
# 1. DeepCSAT_best_model.joblib   (required)
# 2. deepcsat_scaler.joblib       (optional, recommended)
# 3. feature_columns.json         (optional, recommended)
#
# Optional helper file format for feature_columns.json:
# [
#   "channel_name", "category", "Sub-category", "Customer_City",
#   "Product_category", "Item_price", "connected_handling_time",
#   "Agent_name", "Supervisor", "Manager", "Tenure Bucket",
#   "Agent Shift", "response_time", "Item_price_log",
#   "handling_time_log"
# ]
# ============================================================

MODEL_PATH = "DeepCSAT_best_model.joblib"
SCALER_PATH = "deepcsat_scaler.joblib"
FEATURE_PATH = "feature_columns.json"
HISTORY_PATH = "prediction_history.csv"

DEFAULT_FEATURES = [
    "channel_name",
    "category",
    "Sub-category",
    "Customer_City",
    "Product_category",
    "Item_price",
    "connected_handling_time",
    "Agent_name",
    "Supervisor",
    "Manager",
    "Tenure Bucket",
    "Agent Shift",
    "response_time",
    "Item_price_log",
    "handling_time_log",
]

SCALED_NUMERIC_COLS = ["Item_price", "connected_handling_time", "response_time"]


@st.cache(allow_output_mutation=True)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please place your saved DeepCSAT model in the app folder."
        )
    return joblib.load(model_path)


@st.cache(allow_output_mutation=True)
def load_scaler(scaler_path: str):
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


@st.cache
def load_feature_columns(feature_path: str):
    if os.path.exists(feature_path):
        with open(feature_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    return DEFAULT_FEATURES


def ensure_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0
    return out[feature_columns]



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Item_price" in out.columns:
        out["Item_price"] = pd.to_numeric(out["Item_price"], errors="coerce").fillna(0)
    if "connected_handling_time" in out.columns:
        out["connected_handling_time"] = pd.to_numeric(
            out["connected_handling_time"], errors="coerce"
        ).fillna(0)
    if "response_time" in out.columns:
        out["response_time"] = pd.to_numeric(out["response_time"], errors="coerce").fillna(0)

    # Derived features used in the trained notebook
    if "Item_price" in out.columns and "Item_price_log" not in out.columns:
        out["Item_price_log"] = np.log1p(np.clip(out["Item_price"], a_min=0, a_max=None))

    if "connected_handling_time" in out.columns and "handling_time_log" not in out.columns:
        out["handling_time_log"] = np.log1p(
            np.clip(out["connected_handling_time"], a_min=0, a_max=None)
        )

    return out



def apply_scaling(df: pd.DataFrame, scaler) -> pd.DataFrame:
    out = df.copy()
    if scaler is None:
        return out

    cols_to_scale = [c for c in SCALED_NUMERIC_COLS if c in out.columns]
    if cols_to_scale:
        out.loc[:, cols_to_scale] = scaler.transform(out[cols_to_scale])
    return out



def score_band(score: float) -> str:
    if score >= 4.5:
        return "High Satisfaction"
    if score >= 3.5:
        return "Moderate Satisfaction"
    return "At Risk"



def business_insights(row: pd.Series, pred: float) -> list[str]:
    insights = []

    response_time = float(row.get("response_time", 0) or 0)
    handling_time = float(row.get("connected_handling_time", 0) or 0)
    item_price = float(row.get("Item_price", 0) or 0)

    if pred < 3.5:
        insights.append("Predicted satisfaction is low. Consider proactive follow-up or escalation.")
    elif pred < 4.5:
        insights.append("Predicted satisfaction is moderate. Small service improvements may raise CSAT.")
    else:
        insights.append("Predicted satisfaction is high. Current service pattern appears effective.")

    if response_time > 1800:
        insights.append("Response time is high. Faster initial response may improve customer perception.")
    if handling_time > 900:
        insights.append("Handling time is high. Review workflow efficiency and issue complexity.")
    if item_price > 5000 and pred < 4.0:
        insights.append("High-value order with modest satisfaction. Review service quality for premium customers.")

    return insights



def save_history(input_df: pd.DataFrame, preds: np.ndarray):
    history_df = input_df.copy()
    history_df["Predicted_CSAT"] = preds
    history_df["Satisfaction_Band"] = history_df["Predicted_CSAT"].apply(score_band)
    history_df["Prediction_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_header = not os.path.exists(HISTORY_PATH)
    history_df.to_csv(HISTORY_PATH, mode="a", header=write_header, index=False)



def render_history():
    st.subheader("Prediction History")
    if os.path.exists(HISTORY_PATH):
        hist = pd.read_csv(HISTORY_PATH)
        st.dataframe(hist.tail(50), use_container_width=True)

        st.markdown("**Recent Summary**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(hist))
        with col2:
            st.metric("Average Predicted CSAT", f"{hist['Predicted_CSAT'].mean():.2f}")
        with col3:
            at_risk = (hist["Predicted_CSAT"] < 3.5).sum()
            st.metric("At-Risk Cases", int(at_risk))
    else:
        st.info("No prediction history found yet.")


st.set_page_config(page_title="DeepCSAT Predictor", page_icon="📊", layout="wide")

st.title("📊 DeepCSAT – Customer Satisfaction Prediction")
st.write(
    "Use this local Streamlit app to generate ongoing CSAT predictions and operational insights "
    "for customer support interactions."
)

try:
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    feature_columns = load_feature_columns(FEATURE_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("App Options")
    mode = st.radio("Choose prediction mode", ["Single Prediction", "Batch Prediction"])
    st.caption("This app expects the same encoded feature format used during model training.")
    if scaler is None:
        st.warning("Scaler file not found. Predictions will run without numeric scaling fallback.")

if mode == "Single Prediction":
    st.subheader("Single Prediction")
    st.write("Enter encoded feature values below. These should match the preprocessing used during training.")

    c1, c2, c3 = st.columns(3)

    with c1:
        channel_name = st.number_input("channel_name", min_value=0, value=0, step=1)
        category = st.number_input("category", min_value=0, value=0, step=1)
        sub_category = st.number_input("Sub-category", min_value=0, value=0, step=1)
        customer_city = st.number_input("Customer_City", min_value=0, value=0, step=1)
        product_category = st.number_input("Product_category", min_value=0, value=0, step=1)

    with c2:
        item_price = st.number_input("Item_price", min_value=0.0, value=1000.0, step=100.0)
        connected_handling_time = st.number_input(
            "connected_handling_time", min_value=0.0, value=420.0, step=10.0
        )
        response_time = st.number_input("response_time", min_value=0.0, value=300.0, step=10.0)
        agent_name = st.number_input("Agent_name", min_value=0, value=0, step=1)
        supervisor = st.number_input("Supervisor", min_value=0, value=0, step=1)

    with c3:
        manager = st.number_input("Manager", min_value=0, value=0, step=1)
        tenure_bucket = st.number_input("Tenure Bucket", min_value=0, value=0, step=1)
        agent_shift = st.number_input("Agent Shift", min_value=0, value=0, step=1)

    input_df = pd.DataFrame(
        {
            "channel_name": [channel_name],
            "category": [category],
            "Sub-category": [sub_category],
            "Customer_City": [customer_city],
            "Product_category": [product_category],
            "Item_price": [item_price],
            "connected_handling_time": [connected_handling_time],
            "Agent_name": [agent_name],
            "Supervisor": [supervisor],
            "Manager": [manager],
            "Tenure Bucket": [tenure_bucket],
            "Agent Shift": [agent_shift],
            "response_time": [response_time],
        }
    )

    if st.button("Predict CSAT", type="primary"):
        prepared = engineer_features(input_df)
        original_for_insights = prepared.copy()
        prepared = ensure_columns(prepared, feature_columns)
        prepared = apply_scaling(prepared, scaler)

        pred = model.predict(prepared)[0]
        pred = float(np.clip(pred, 1, 5))

        st.success(f"Predicted CSAT Score: {pred:.2f}")
        st.metric("Satisfaction Band", score_band(pred))

        st.markdown("### Operational Insights")
        for insight in business_insights(original_for_insights.iloc[0], pred):
            st.write(f"- {insight}")

        save_history(original_for_insights, np.array([pred]))

else:
    st.subheader("Batch Prediction")
    st.write(
        "Upload a CSV file containing the encoded feature columns used by the model. "
        "The app will generate predicted CSAT scores and downloadable results."
    )

    st.markdown("**Expected core input columns**")
    st.code(
        ", ".join(
            [
                "channel_name",
                "category",
                "Sub-category",
                "Customer_City",
                "Product_category",
                "Item_price",
                "connected_handling_time",
                "Agent_name",
                "Supervisor",
                "Manager",
                "Tenure Bucket",
                "Agent Shift",
                "response_time",
            ]
        )
    )

    uploaded = st.file_uploader("Upload input CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            prepared = engineer_features(batch_df)
            original_for_insights = prepared.copy()
            prepared = ensure_columns(prepared, feature_columns)
            prepared = apply_scaling(prepared, scaler)

            preds = model.predict(prepared)
            preds = np.clip(preds, 1, 5)

            result_df = original_for_insights.copy()
            result_df["Predicted_CSAT"] = preds
            result_df["Satisfaction_Band"] = result_df["Predicted_CSAT"].apply(score_band)

            st.success("Batch prediction completed.")
            st.dataframe(result_df.head(50), use_container_width=True)

            save_history(original_for_insights, preds)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_bytes,
                file_name="deepcsat_batch_predictions.csv",
                mime="text/csv",
            )

render_history()

st.markdown("---")
st.markdown("### Local Deployment Notes")
st.code(
    "pip install streamlit pandas numpy scikit-learn joblib\n"
    "streamlit run deepcsat_streamlit_app.py",
    language="bash",
)
st.caption(
    "For best results, keep the saved model, optional scaler, and this app file in the same local project folder."
)
