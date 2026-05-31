import pandas as pd
from transformers import pipeline
import torch
import os

import sys
INPUT = sys.argv[1] if len(sys.argv) > 1 else "skoki.parquet"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else INPUT.replace(".parquet", "_sentiment.csv")

print(f"CUDA: {torch.cuda.is_available()}")
df = pd.read_parquet(INPUT)
print(f"Člankov: {len(df)}")

# Pripravi tekst: naslov (lead) + prvih 400 znakov telesa
def prep_text(row):
    lead = str(row["lead"] or "")
    body = str(row["body"] or "")
    combined = lead + " " + body
    return combined[:512].strip()

df["text"] = df.apply(prep_text, axis=1)

print("Nalagam model...")
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=0,
    batch_size=64,
    truncation=True,
    max_length=512,
)

print("Zaganjam inferenco...")
texts = df["text"].tolist()

results = []
batch_size = 64
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    out = classifier(batch)
    results.extend(out)
    if i % 256 == 0:
        print(f"  {i}/{len(texts)}...")

# Pretvori label v numerično vrednost
def to_score(r):
    label = r["label"].lower()
    score = r["score"]
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0

df["sentiment_raw"] = [to_score(r) for r in results]
df["sentiment_label"] = [r["label"] for r in results]

# Shrani
cols = ["id", "url", "date", "keywords", "mentions", "lead", "sentiment_raw", "sentiment_label"]
df[cols].to_csv(OUTPUT, index=False)
print(f"\nShranjeno v {OUTPUT}")

# Hiter pregled
print(f"\nPorazdelitev labelov:")
print(df["sentiment_label"].value_counts())
print(f"\nPovprečni sentiment: {df['sentiment_raw'].mean():.3f}")
print(f"Min: {df['sentiment_raw'].min():.3f}, Max: {df['sentiment_raw'].max():.3f}")
