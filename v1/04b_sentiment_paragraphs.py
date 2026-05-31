"""
04b_sentiment_paragraphs.py — Paragraph-level sentiment analiza

Za razliko od 04_sentiment.py (ki dela na lead-u) ta skript razdeli vsakega
članka na odstavke in požene sentiment na vsakem posebej.

Predpogoj: parquet datoteke iz pipeline 01→02→08
  - skoki.parquet, kolesarstvo.parquet, ipd.

Uporaba:
  python3 04b_sentiment_paragraphs.py skoki.parquet skoki_sentiment_paragraphs.csv
  python3 04b_sentiment_paragraphs.py  # privzeto: skoki.parquet

Output CSV stolpci:
  id, url, date, paragraph_idx, paragraph_text, sentiment_raw, sentiment_label
"""

import sys
import os
import pandas as pd
import torch
from transformers import pipeline

# ── Parametri ────────────────────────────────────────────────────────────────
INPUT  = sys.argv[1] if len(sys.argv) > 1 else "skoki.parquet"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else INPUT.replace(".parquet", "_sentiment_paragraphs.csv")

MIN_WORDS    = 10   # odstavki krajši od tega se preskočijo
BATCH_SIZE   = 64
MAX_CHARS    = 512  # model limit (v besedah ~90, v znakih varno 512)

# ── Filtriranje neinformativnih odstavkov ─────────────────────────────────────
SKIP_PATTERNS = [
    # Tipični "boilerplate" odstavki na MMC
    "preberite tudi",
    "oglejte si",
    "foto:",
    "video:",
    "©",
    "vse pravice pridržane",
    "rtv slovenija",
]

def is_informative(text: str) -> bool:
    """Vrne False za prekratke ali boilerplate odstavke."""
    t = text.strip()
    if len(t.split()) < MIN_WORDS:
        return False
    tl = t.lower()
    if any(pat in tl for pat in SKIP_PATTERNS):
        return False
    return True

# ── Naloži podatke ────────────────────────────────────────────────────────────
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Berem {INPUT}...")
df = pd.read_parquet(INPUT)
print(f"Člankov: {len(df)}")

# Preverimo da imamo 'paragraphs' stolpec
if "paragraphs" not in df.columns:
    raise ValueError(
        "Parquet nima stolpca 'paragraphs'. "
        "Preveri 01_yaml_to_parquet.py — paragraphs morajo biti shranjeni kot seznam."
    )

# ── Razstavi na odstavke ──────────────────────────────────────────────────────
rows = []
for _, article in df.iterrows():
    paragraphs = article.get("paragraphs")

    # Parquet vrne numpy array, Python list, string ali None
    if paragraphs is None:
        paragraphs = []
    elif isinstance(paragraphs, str):
        paragraphs = [p.strip() for p in paragraphs.split("\n") if p.strip()]
    else:
        paragraphs = list(paragraphs)  # numpy array -> list

    for idx, para in enumerate(paragraphs):
        para = str(para).strip()
        if not is_informative(para):
            continue
        rows.append({
            "id":             article["id"],
            "url":            article.get("url", ""),
            "date":           article.get("date"),
            "paragraph_idx":  idx,
            "paragraph_text": para[:MAX_CHARS],  # truncate pred inferenčo
        })

para_df = pd.DataFrame(rows)
print(f"Informativnih odstavkov: {len(para_df)} "
      f"(povp. {len(para_df)/len(df):.1f} na članek)")

if len(para_df) == 0:
    raise ValueError(
        "Ni odstavkov. Preveri ali parquet vsebuje stolpec 'paragraphs' z vsebino."
    )

# ── Model ─────────────────────────────────────────────────────────────────────
print("Nalagam model...")
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=0 if torch.cuda.is_available() else -1,
    batch_size=BATCH_SIZE,
    truncation=True,
    max_length=512,
)

# ── Inferenca ─────────────────────────────────────────────────────────────────
print("Zaganjam inferenco...")
texts = para_df["paragraph_text"].tolist()
results = []
total = len(texts)
bar_width = 40

for i in range(0, total, BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    out = classifier(batch)
    results.extend(out)

    done = min(i + BATCH_SIZE, total)
    filled = int(bar_width * done / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = 100 * done / total
    print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}", end="", flush=True)

print()  # nova vrstica po koncu
# ── Pretvori label → numerična vrednost (enako kot 04_sentiment.py) ───────────
def to_score(r):
    label = r["label"].lower()
    score = r["score"]
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0

para_df["sentiment_raw"]   = [to_score(r)         for r in results]
para_df["sentiment_label"] = [r["label"].lower()   for r in results]

# ── Shrani ────────────────────────────────────────────────────────────────────
cols = ["id", "url", "date", "paragraph_idx", "paragraph_text",
        "sentiment_raw", "sentiment_label"]
para_df[cols].to_csv(OUTPUT, index=False)
print(f"\nShranjeno v {OUTPUT}")

# ── Hiter pregled ─────────────────────────────────────────────────────────────
print(f"\nPorazdelitev labelov:")
print(para_df["sentiment_label"].value_counts())
print(f"\nPovprečni sentiment: {para_df['sentiment_raw'].mean():.3f}")
print(f"Min: {para_df['sentiment_raw'].min():.3f}, Max: {para_df['sentiment_raw'].max():.3f}")

# Primerjava: paragraph vs article distribucija
n_pos_p = (para_df["sentiment_label"] == "positive").mean() * 100
n_neg_p = (para_df["sentiment_label"] == "negative").mean() * 100
print(f"\nParagraph level — pozitivnih: {n_pos_p:.1f}%, negativnih: {n_neg_p:.1f}%")
