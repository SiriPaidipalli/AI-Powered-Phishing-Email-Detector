import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("AI-Powered Phishing Email Detector")

@st.cache_resource
def load_bundle():
    bundle = joblib.load("models/baseline_lr.joblib")
    return bundle["vectorizer"], bundle["model"], bundle["threshold"]

vec, model, thr = load_bundle()

st.markdown("Enter a subject and body. The model prioritizes catching phishing (higher recall).")

subj = st.text_input("Email subject")
body = st.text_area("Email body", height=180)

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict"):
        text = f"subject: {subj} body: {body}"
        X = vec.transform([text])
        p = model.predict_proba(X)[0, 1]
        label = "Phishing" if p >= thr else "Legitimate"
        st.subheader(f"Prediction: {label}")
        st.write(f"Score: {p:.3f}  |  Threshold: {thr:.3f}")

with col2:
    st.caption("Why flagged (top tokens)")
    if subj or body:
        text = f"subject: {subj} body: {body}"
        X = vec.transform([text])
        # feature importances for LR on TF-IDF
        try:
            coefs = model.coef_[0]
            idxs = X.nonzero()[1]
            weights = [(vec.get_feature_names_out()[i], float(coefs[i] * X[0, i])) for i in idxs]
            top = sorted(weights, key=lambda t: t[1], reverse=True)[:8]
            if top:
                for token, w in top:
                    st.write(f"{token} — {w:.4f}")
            else:
                st.write("No salient tokens.")
        except Exception:
            st.write("Explanation not available for this model type.")

st.divider()
st.caption("Batch mode: upload CSV with columns `subject, body` to get predictions.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    import pandas as pd
    df = pd.read_csv(uploaded)
    if not {"subject", "body"}.issubset(df.columns):
        st.error("CSV must have columns: subject, body")
    else:
        texts = ("subject: " + df["subject"].astype(str) + " body: " + df["body"].astype(str)).tolist()
        X = vec.transform(texts)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= thr).astype(int)
        out = df.copy()
        out["score"] = probs
        out["prediction"] = np.where(preds == 1, "Phishing", "Legitimate")
        st.success(f"Processed {len(out)} rows.")
        st.dataframe(out.head(20))
        # allow download
        out.to_csv("predictions.csv", index=False)
        st.download_button("Download results CSV", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
