"""
Pripravi SQLite bazo za oznacevalnik iz sport_clanki.parquet.

Lokalna uporaba:
  python3 prepare.py --input ../v2/sport_clanki.parquet
  python3 prepare.py --input ../v2/sport_clanki.parquet --existing-labels /pot/do/oznake.csv

Docker: nastavi env spremenljivke PARQUET_PATH, DB_PATH, EXISTING_LABELS.
"""

import argparse
import os
import random
import sqlite3
import sys

import pandas as pd

SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis",
]

parser = argparse.ArgumentParser()
parser.add_argument("--input",           default=os.environ.get("PARQUET_PATH", "../v2/sport_clanki.parquet"))
parser.add_argument("--existing-labels", default=os.environ.get("EXISTING_LABELS", ""))
parser.add_argument("--seed",            type=int, default=42)
parser.add_argument("--db",              default=os.environ.get("DB_PATH", "articles.db"))
args = parser.parse_args()

if not os.path.exists(args.input):
    print(f"NAPAKA: {args.input} ne obstaja.")
    sys.exit(1)

# ── Naloži že označene ID-je ──────────────────────────────────────────────────
existing_ids = set()
if args.existing_labels and os.path.exists(args.existing_labels):
    existing = pd.read_csv(args.existing_labels)
    existing_ids = set(existing["id"].astype(str).tolist())
    print(f"Obstojecih oznak (preskoci): {len(existing_ids)}")

# ── Naloži parquet ────────────────────────────────────────────────────────────
print(f"Berem {args.input}...")
df = pd.read_parquet(args.input)
df["id"] = df["id"].astype(str)

# Filtriraj na 12 sportov
df = df[df["sport"].isin(SPORTS)].copy()
print(f"Sportnih clankov (12 sportov): {len(df)}")

# Izloci ze oznacene
df = df[~df["id"].isin(existing_ids)].copy()
print(f"Po izlocitvi ze oznacenih: {len(df)}")

# ── Pripravi besedilo ──────────────────────────────────────────────────────────
def build_body(row):
    lead = (row.get("lead") or "").strip()
    paragraphs = row.get("paragraphs")
    if paragraphs is None:
        paragraphs = []
    elif isinstance(paragraphs, str):
        paragraphs = [p.strip() for p in paragraphs.split("\n") if p.strip()]
    else:
        paragraphs = [str(p).strip() for p in paragraphs if p]
    parts = ([lead] if lead else []) + [p for p in paragraphs if p and p != lead]
    return "\n\n".join(parts)

valid = []
for _, row in df.iterrows():
    body = build_body(row)
    if len(body.split()) < 30:
        continue
    valid.append({
        "id":    row["id"],
        "url":   str(row.get("url") or ""),
        "date":  str(row.get("date") or "")[:10],
        "title": str(row.get("title") or "").strip(),
        "lead":  str(row.get("lead") or "").strip(),
        "body":  body,
        "sport": str(row.get("sport") or ""),
    })

print(f"Clankov z vsebino: {len(valid)}")

# Premesaj
random.seed(args.seed)
random.shuffle(valid)

# ── Ustvari bazo ──────────────────────────────────────────────────────────────
if os.path.exists(args.db):
    os.remove(args.db)

conn = sqlite3.connect(args.db)
conn.executescript("""
    CREATE TABLE articles (
        id         TEXT PRIMARY KEY,
        url        TEXT,
        date       TEXT,
        title      TEXT,
        lead       TEXT,
        body       TEXT,
        sport      TEXT,
        sort_order INTEGER
    );
    CREATE TABLE labels (
        article_id TEXT PRIMARY KEY,
        label      TEXT,
        labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

for i, art in enumerate(valid):
    conn.execute(
        "INSERT INTO articles VALUES (?,?,?,?,?,?,?,?)",
        (art["id"], art["url"], art["date"], art["title"],
         art["lead"], art["body"], art["sport"], i),
    )

conn.commit()
conn.close()

print(f"Shranjeno v {args.db} ({len(valid)} clankov, brez zgornje meje)")
