import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def train_baseline(input_csv, model_path):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} not found. Run data_prep.py first.")
    df = pd.read_csv(input_csv)
    X, y = df["text"], df["label"]

    # Split train/dev/test (80/10/10)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # TF-IDF features
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    Xtr = vec.fit_transform(X_train)

    # Logistic Regression (balanced)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, y_train)

    # threshold tuning for high phishing recall
    Xdv = vec.transform(X_dev)
    probs = clf.predict_proba(Xdv)[:,1]
    prec, rec, thr = precision_recall_curve(y_dev, probs)
    best_i = rec.argmax()
    best_thr = thr[best_i-1] if best_i>0 else 0.5

    # Evaluate on test
    Xte = vec.transform(X_test)
    test_probs = clf.predict_proba(Xte)[:,1]
    y_pred = (test_probs >= best_thr).astype(int)

    print("\nClassification Report (threshold tuned for recall)\n")
    print(classification_report(y_test, y_pred, target_names=["legit","phish"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    ap = average_precision_score(y_test, test_probs)
    print(f"Average precision: {ap:.3f}")
    print(f"Chosen threshold: {best_thr:.3f}")

    # save model bundle
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": clf, "threshold": best_thr}, model_path)
    print(f"✅ Model saved → {model_path}")

    # PR curve
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Dev set Precision-Recall")
    plt.grid(True)
    plt.savefig("../app/dev_pr_curve.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    train_baseline("../data/processed/emails.csv", "../models/baseline_lr.joblib")
