import yaml
import pandas as pd
import sys
import os

INPUT = sys.argv[1] if len(sys.argv) > 1 else "mmc-100.yaml"
OUTPUT = INPUT.replace(".yaml", ".parquet")

file_size = os.path.getsize(INPUT)
print(f"Berem {INPUT}... ({file_size // (1024*1024)} MB)")
print("(Branje v pomnilnik, to traja trenutek...)")

with open(INPUT, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

total = len(data)
print(f"Prebranih {total} dokumentov. Pretvarjam v DataFrame...")

rows = []
bar_width = 40

for i, doc in enumerate(data):
    lead = doc.get("lead", "") or ""
    paragraphs = doc.get("paragraphs", []) or []
    body = lead + " " + " ".join(paragraphs)
    rows.append({
        "id":         doc.get("id", ""),
        "url":        doc.get("url", ""),
        "date":       doc.get("date", ""),
        "topics":     doc.get("topics", ""),
        "keywords":   ", ".join(doc.get("keywords", []) or []),
        "mentions":   ", ".join(doc.get("mention", []) or []),
        "title":      doc.get("title", "") or "",
        "n_comments": doc.get("n_comments", 0) or 0,
        "lead":       lead,
        "body":       body.strip(),
        "paragraphs": paragraphs,
    })

    if i % 100 == 0 or i == total - 1:
        filled = int(bar_width * (i + 1) / total)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = 100 * (i + 1) / total
        print(f"\r  [{bar}] {pct:5.1f}%  {i+1}/{total}", end="", flush=True)

print()

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print(df.dtypes)
print(f"\nVzorec:")
print(df[["date", "topics", "lead"]].head(3))
print(f"\nTopics distribucija:")
print(df["topics"].value_counts().head(10))

df.to_parquet(OUTPUT, index=False)
print(f"\nShranjeno v {OUTPUT} ({os.path.getsize(OUTPUT)//1024} KB)")
