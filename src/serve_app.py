from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app_inference import (
    MAX_BATCH_ROWS,
    MAX_UPLOAD_BYTES,
    InputValidationError,
    analyze_batch,
    analyze_message,
    safe_spreadsheet_text,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "models" / "baseline_lr.joblib"
CATEGORY_LABELS = {
    "urgency": "Urgency",
    "credentials": "Credentials and account verification",
    "payment": "Payment, invoice, or banking",
    "call_to_action": "Suspicious call to action",
    "security_alert": "Impersonation or security alert",
    "attachment_execution": "Attachment execution",
    "secrecy": "Secrecy or process bypass",
    "insecure_url": "Unencrypted URL",
    "ip_address_url": "IP-address URL",
    "obfuscated_url": "Obfuscated URL",
    "url_domain_mismatch": "Visible and destination domain mismatch",
}


st.set_page_config(page_title="Phishing Email Analysis", layout="wide")
st.title("Phishing Email Analysis")
st.caption(
    "Detection and triage prototype. Results should support, not replace, secure email "
    "controls and analyst review."
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def show_evidence_value(value) -> None:
    if isinstance(value, dict):
        visible = value.get("visible_domain", "unknown")
        destination = value.get("destination_domain", "unknown")
        st.markdown(f"- Visible: `{visible}` → destination: `{destination}`")
    else:
        st.markdown(f"- `{value}`")


def show_single_result(result) -> None:
    risk = result["risk"]
    analysis = result["security_analysis"]
    top_left, top_middle, top_right = st.columns(3)
    top_left.metric("Risk level", risk["risk_level"])
    top_middle.metric("Risk score", f"{risk['risk_points']} / 8")
    top_right.metric("Model classification", result["model_prediction"])

    model_left, model_right = st.columns(2)
    model_left.metric("ML phishing probability", f"{result['ml_probability']:.1%}")
    model_right.metric("ML classification threshold", f"{result['model_threshold']:.1%}")
    st.info("The risk score is a deterministic triage score, not a calibrated phishing probability.")

    st.subheader("Why was this flagged?")
    st.markdown("#### Rule-based security evidence")
    if result["indicator_groups"]:
        for category, indicators in result["indicator_groups"].items():
            st.markdown(f"**{CATEGORY_LABELS.get(category, category.replace('_', ' ').title())}**")
            for indicator in indicators:
                st.caption(indicator["rule"])
                for evidence in indicator["evidence"]:
                    show_evidence_value(evidence)
    else:
        st.write("No deterministic security indicators matched.")

    st.markdown("#### Extracted message artifacts")
    artifact_left, artifact_right = st.columns(2)
    with artifact_left:
        st.markdown("**URLs**")
        if analysis["urls"]:
            for url in analysis["urls"]:
                st.code(url, language=None)
        else:
            st.write("None found.")
    with artifact_right:
        st.markdown("**Email addresses**")
        if analysis["email_addresses"]:
            for address in analysis["email_addresses"]:
                st.code(address, language=None)
        else:
            st.write("None found.")
    st.caption(
        "Indicator categories: "
        + (", ".join(analysis["categories"]) if analysis["categories"] else "None")
    )

    st.markdown("#### Model evidence")
    st.caption(
        "These are TF-IDF feature contributions from the logistic-regression model. "
        "They describe model behavior and are not causal explanations."
    )
    phishing_column, legitimate_column = st.columns(2)
    with phishing_column:
        st.markdown("**Strongest features toward phishing**")
        features = result["model_evidence"]["toward_phishing"]
        if features:
            for item in features:
                st.markdown(f"- `{item['feature']}`: `{item['contribution']:.4f}`")
        else:
            st.write("No positive feature contributions in this message.")
    with legitimate_column:
        st.markdown("**Strongest features toward legitimate**")
        features = result["model_evidence"]["toward_legitimate"]
        if features:
            for item in features:
                st.markdown(f"- `{item['feature']}`: `{item['contribution']:.4f}`")
        else:
            st.write("No negative feature contributions in this message.")


try:
    bundle = load_bundle()
except Exception as error:
    st.error(f"The trained baseline model could not be loaded: {error}")
    st.stop()


st.subheader("Single email")
with st.form("single_email_form"):
    subject = st.text_input("Subject")
    body = st.text_area("Body", height=220)
    submitted = st.form_submit_button("Analyze email", type="primary")

if submitted:
    try:
        st.session_state["single_result"] = analyze_message(bundle, subject, body)
    except InputValidationError as error:
        st.error(str(error))
        st.session_state.pop("single_result", None)

if "single_result" in st.session_state:
    show_single_result(st.session_state["single_result"])


st.divider()
st.subheader("Batch CSV")
st.caption(
    f"Upload a CSV with subject and body columns. Batches are limited to {MAX_BATCH_ROWS} rows."
)
uploaded = st.file_uploader("Email CSV", type=["csv"])
if uploaded is not None:
    try:
        if uploaded.size > MAX_UPLOAD_BYTES:
            raise InputValidationError(
                f"CSV is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
            )
        frame = pd.read_csv(uploaded)
        if not {"subject", "body"}.issubset(frame.columns):
            raise InputValidationError("CSV must contain subject and body columns.")
        records = frame[["subject", "body"]].to_dict(orient="records")
        results = analyze_batch(bundle, records)
        output = pd.DataFrame(results)
        display = output.copy()
        display["phishing_probability"] = display["phishing_probability"].map(
            lambda value: f"{value:.1%}"
        )
        st.success(f"Analyzed {len(output)} emails.")
        st.dataframe(display, use_container_width=True, hide_index=True)
        download = output.copy()
        download["subject"] = download["subject"].map(safe_spreadsheet_text)
        st.download_button(
            "Download analysis CSV",
            data=download.to_csv(index=False),
            file_name="phishing_analysis.csv",
            mime="text/csv",
        )
    except (
        InputValidationError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(str(error))
