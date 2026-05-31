"""
03_validacija.py

Primerja oznake iz oznacevalnika z napovedmi treh modelov.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report

LABELS_CSV = "/home/miha/Downloads/oznake.csv"

MODELS = {
    "lxyuan":           "sentiment_distilbert_base_multilingual_cased_sentiments_student.parquet",
    "xlm-multilingual": "sentiment_xlm_roberta_base_sentiment_multilingual.parquet",
    "twitter-xlm":      "sentiment_twitter_xlm_roberta_base_sentiment.parquet",
}

# ── Naloži ročne oznake ────────────────────────────────────────────────────────
labels = pd.read_csv(LABELS_CSV)
labels = labels[["id", "label"]].rename(columns={"label": "true"})
labels["id"] = labels["id"].astype(str)
print(f"Ročnih oznak: {len(labels)}")
print(labels["true"].value_counts().to_string())

# ── Primerjaj vsak model ───────────────────────────────────────────────────────
for model_name, parquet_file in MODELS.items():
    try:
        df = pd.read_parquet(parquet_file)[["id", "sentiment_label"]].copy()
        df["id"] = df["id"].astype(str)
        merged = labels.merge(df, on="id", how="inner")

        if len(merged) == 0:
            print(f"\n{model_name}: ni ujemanj po ID-ju")
            continue

        y_true = merged["true"]
        y_pred = merged["sentiment_label"]

        kappa = cohen_kappa_score(y_true, y_pred)

        print(f"\n{'='*50}")
        print(f"Model: {model_name}  (n={len(merged)})")
        print(f"Cohen κ: {kappa:.3f}")
        print(classification_report(y_true, y_pred, zero_division=0))

    except FileNotFoundError:
        print(f"\n{model_name}: datoteka {parquet_file} ne obstaja")
