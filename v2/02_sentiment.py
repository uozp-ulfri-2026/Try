"""
02_sentiment.py

Sentiment analiza na nivoju odstavkov za izbranih 12 sportov.
Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student

Izhod: sentiment.parquet
  id, url, date, title, lead, sport, sentiment_raw, sentiment_label, n_paragraphs
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

# ── Argumenti ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="sport_clanki.parquet")
parser.add_argument("--output", default=None)
parser.add_argument("--batch",  type=int, default=64)
parser.add_argument("--model",  default="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                    help="HuggingFace model ID")
args = parser.parse_args()

# Ime outputa iz modela ce ni podano
if args.output is None:
    slug = args.model.split("/")[-1].replace("-", "_")
    args.output = f"sentiment_{slug}.parquet"

print(f"Model:  {args.model}")
print(f"Output: {args.output}")

SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis",
]

MIN_WORDS   = 10
MAX_CHARS   = 2000
SKIP_PATTERNS = ["preberite tudi", "oglejte si", "foto:", "video:", "©",
                 "vse pravice pridržane", "rtv slovenija"]

# ── Naloži podatke ─────────────────────────────────────────────────────────────
print(f"Berem {args.input}...")
df = pd.read_parquet(args.input)
df = df[df["sport"].isin(SPORTS)].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).reset_index(drop=True)
print(f"Clankov za analizo: {len(df)}")

# ── Razstavi na odstavke ───────────────────────────────────────────────────────
def is_informative(text):
    t = text.strip()
    words = t.split()
    if len(words) < MIN_WORDS:
        return False
    tl = t.lower()
    if any(p in tl for p in SKIP_PATTERNS):
        return False
    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    if caps_words / len(words) > 0.25:
        return False
    return True

para_rows = []
for _, row in df.iterrows():
    art_id = row["id"]
    lead   = (row.get("lead") or "").strip()

    paragraphs = row.get("paragraphs")
    if paragraphs is None:
        paragraphs = []
    elif isinstance(paragraphs, str):
        paragraphs = [p.strip() for p in paragraphs.split("\n") if p.strip()]
    else:
        paragraphs = list(paragraphs)

    # Filtriraj informativne odstavke
    informative = [(idx, str(p).strip()) for idx, p in enumerate(paragraphs)
                   if is_informative(str(p))]

    # Ce ni odstavkov, vzami lead kot edini odstavek
    if not informative and lead:
        informative = [(0, lead)]

    n = len(informative)

    # Lead (vedno loceno)
    if lead and is_informative(lead):
        para_rows.append({
            "article_id":     art_id,
            "para_type":      "lead",
            "para_order":     -1,
            "paragraph_text": lead[:MAX_CHARS],
        })

    # Odstavki z tipi: first / middle / last
    for i, (orig_idx, text) in enumerate(informative):
        if n <= 1:
            ptype = "first"
        elif i == 0:
            ptype = "first"
        elif i == n - 1:
            ptype = "last"
        else:
            ptype = "middle"
        para_rows.append({
            "article_id":     art_id,
            "para_type":      ptype,
            "para_order":     orig_idx,
            "paragraph_text": text[:MAX_CHARS],
        })

para_df = pd.DataFrame(para_rows)
print(f"Odstavkov za analizo: {len(para_df)} "
      f"(povp. {len(para_df)/len(df):.1f} na clanek)")
print(para_df["para_type"].value_counts().to_string())

# ── Model ──────────────────────────────────────────────────────────────────────
device = 0 if torch.cuda.is_available() else -1
print(f"Nalagam model... (device: {'cuda' if device == 0 else 'cpu'})")
classifier = pipeline(
    "sentiment-analysis",
    model=args.model,
    device=device,
    batch_size=args.batch,
    truncation=True,
    max_length=512,
)

# ── Inferenca ──────────────────────────────────────────────────────────────────
print("Zaganjam inferenco...")
texts = para_df["paragraph_text"].tolist()
results = []
total = len(texts)
bar_width = 40

for i in range(0, total, args.batch):
    batch = texts[i:i + args.batch]
    results.extend(classifier(batch))
    done = min(i + args.batch, total)
    filled = int(bar_width * done / total)
    print(f"\r  [{'█'*filled}{'░'*(bar_width-filled)}] {100*done/total:5.1f}%  {done}/{total}",
          end="", flush=True)

print()

# ── Numerična vrednost ─────────────────────────────────────────────────────────
def to_score(r):
    label = r["label"].lower()
    score = r["score"]
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    return 0.0

para_df["sentiment_raw"]   = [to_score(r) for r in results]
para_df["sentiment_label"] = [r["label"].lower() for r in results]

# ── Shrani paragraph-level podatke ────────────────────────────────────────────
para_output = args.output.replace(".parquet", "_paragraphs.parquet")
para_df[["article_id", "para_type", "para_order", "paragraph_text",
         "sentiment_raw", "sentiment_label"]].to_parquet(para_output, index=False)
print(f"Paragraphs shranjeni: {para_output}")

# ── Agregacija na nivo clanka ──────────────────────────────────────────────────
def majority_label(s):
    return s.value_counts().idxmax()

agg = para_df.groupby("article_id").agg(
    sentiment_raw   = ("sentiment_raw",   "mean"),
    sentiment_label = ("sentiment_label", majority_label),
    n_paragraphs    = ("para_order",   "count"),
).reset_index().rename(columns={"article_id": "id"})

# Spoji z originalnimi metapodatki
result = df[["id", "url", "date", "title", "lead", "sport", "confidence"]].merge(
    agg, on="id", how="inner"
)

# ── Shrani ─────────────────────────────────────────────────────────────────────
result.to_parquet(args.output, index=False)
print(f"\nShranjeno: {args.output}  ({len(result)} clankov)")

# ── Statistika ─────────────────────────────────────────────────────────────────
print("\nPorazdelitev labelov:")
print(result["sentiment_label"].value_counts().to_string())
print(f"\nPovp. sentiment_raw: {result['sentiment_raw'].mean():.3f}")
print(f"Povp. odstavkov/clanek: {result['n_paragraphs'].mean():.1f}")
print("\nPo sportu (povp. sentiment):")
print(result.groupby("sport")["sentiment_raw"].agg(["mean","count"]).round(3).to_string())
