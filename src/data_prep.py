import os
import re
import pandas as pd
from bs4 import BeautifulSoup

URL_PAT = re.compile(r'https?://\S+')
EMAIL_PAT = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # strip HTML → lower → normalize
    text = BeautifulSoup(text, "lxml").get_text(" ")
    text = text.lower()
    text = URL_PAT.sub(" URL ", text)
    text = EMAIL_PAT.sub(" EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_dataset(input_csv: str, output_csv: str) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}\n"
                                f"Create it at data/raw/combined.csv with columns: subject,body,label")
    df = pd.read_csv(input_csv)

    required = {"subject", "body", "label"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV must contain columns {required}. Missing: {missing}")

    # clean and combine
    df["subject_clean"] = df["subject"].map(clean_text)
    df["body_clean"]    = df["body"].map(clean_text)
    df["text"] = "subject: " + df["subject_clean"] + " body: " + df["body_clean"]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    # simple stats
    phish = int((df["label"] == 1).sum())
    legit = int((df["label"] == 0).sum())
    print(f"✅ Cleaned data saved → {output_csv}")
    print(f"   Samples: {len(df)} | phish={phish} | legit={legit}")

if __name__ == "__main__":
    preprocess_dataset("../data/raw/combined.csv", "../data/processed/emails.csv")
