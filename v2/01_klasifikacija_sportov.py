"""
01_klasifikacija_sportov.py

Klasificira sportne clanke iz mmc.yaml po sportu z multilingual sentence embeddings.
Model: paraphrase-multilingual-MiniLM-L12-v2

Izhod: sport_clanki.parquet
  id, url, date, title, lead, body, paragraphs, keywords, sport, confidence
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

# ── Argumenti ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",     default="../mmc.yaml")
parser.add_argument("--output",    default="sport_clanki.parquet")
parser.add_argument("--threshold", type=float, default=0.25,
                    help="Min. confidence za dodelitev sporta; pod tem = 'Drugo'")
parser.add_argument("--batch",     type=int, default=256)
args = parser.parse_args()

# ── Sidra po sportu ────────────────────────────────────────────────────────────
SPORT_ANCHORS = {
    "Nogomet":           "nogomet gol zadetek vratar branilec napadalec vezist liga premier league bundesliga serie a la liga uefa fifa champions league euro svetovno prvenstvo reprezentanca clubs",
    "Košarka":           "košarka koš trojka NBA Euroliga FIBA košarkar točke skoki podaje finale liga prvakov",
    "Kolesarstvo":       "kolesarstvo kolesar etapa Tour de France Giro d Italia Vuelta UCI sprint gorski cilj kronometer dirka po Sloveniji",
    "Smučarski skoki":   "smučarski skoki skakalec skakalnica Prevc Planica Willingen Vikersund Oberstdorf Innsbruck turnej štirih svetovni pokal Raw Air letalni kilometer",
    "Odbojka":           "odbojka odbojkar set as blok napad sprejem libero CEV FIVB liga prvakov odbojkarski klub",
    "Rokomet":           "rokomet rokometaš gol vratar EHF svetovno prvenstvo evropsko prvenstvo liga prvakov rokometni klub",
    "Alpsko smučanje":   "alpsko smučanje smučar slalom veleslalom superveleslalom smuk kombinacija Aare Kitzbühel Kranjska Gora FIS svetovni pokal",
    "Hokej na ledu":     "hokej hokejist gol vratar NHL IIHF plošček ledena dvorana powerplay liga prvakov svetovno prvenstvo",
    "Atletika":          "atletika sprint tek na 100m skok v daljino suvanje krogle met kladiva maraton diamantna liga World Athletics stadion atletinjski rekord",
    "Tenis":             "tenis ATP WTA Roland Garros Wimbledon US Open Australian Open grand slam set game servis forhend bekhend reket teniški igrišče",
    "Motošport":         "formula 1 F1 MotoGP WRC dirkalnik dirka pit stop pole position zavoji motor reli rallycross",
    "Športno plezanje":  "plezanje plezalec balvansko težavnostno hitrostno IFSC stena smer olimpijske igre kombinacija",
    "Plavanje":          "plavanje plavalec prsno hrbtno delfin prosto štafeta World Aquatics olimpijski bazen sekunde",
    "Biatlon":           "biatlon biatlonec streljanje tek na smučeh IBU Pokljuka sprint zasledovanje skupinski start kazen strelišče Bø Fourcade",
    "Judo":              "judo judoist ippon waza-ari IJF tatami kategorija svetovno prvenstvo olimpijske igre pas randori",
    "Smučarski tek":     "smučarski tek smučar klasična tehnika prosta tehnika FIS Tour de Ski sprint skiathlon maraton na smučeh",
    "Kajak in kanu":     "kajak kanu veslaška divja voda mirna voda ICF slalom sprint olimpijske igre čoln veslanje",
    "Borilni športi":    "boks MMA kickboxing karate taekwondo wrestling borba ring kategorija nokavt UFC pas prvak",
    "Namizni tenis":     "namizni tenis ping pong ITTF miza žogica World Table Tennis dvorana",
    "Golf":              "golf fairway par birdie eagle bogey PGA European Tour Ryder Cup Masters igrišče",
    "Badminton":         "badminton BWF reket BWF World Tour shuttle",
    "Strelstvo":         "strelstvo strelec ISSF pištola puška skeet trap streljanje",
    "Gimnastika":        "gimnastika gimnastičar FIG obroč bradlja greda tla preskok ritmična gimnastika akrobatika",
    "Triatlon":          "triatlon triatlonec ITU ironman triatlonski T1 T2 menjalna cona",
    "Konjeništvo":       "konjeništvo konj jezdec FEI dresura preskakovanje ovir hipodrom",
    "Veslanje":          "veslanje veslaš čoln World Rowing regata dvojec četverec osmica ergometer",
    "Lokostrelstvo":     "lokostrelstvo lokostrelec lok puščica World Archery recurve compound",
    "Squash":            "squash PSA squash igrišče squash reket squash žogica",
    "Curling":           "curling kamen metla skip WCF led dvorana button hog line free guard",
    "Ameriški nogomet":  "ameriški nogomet NFL Super Bowl touchdown quarterback receiver lineman American football",
    "Drugo":             "šport rezultat tekma tekmovanje zmaga poraz reprezentanca klub trener športnik",
}

SPORT_NAMES = list(SPORT_ANCHORS.keys())
ANCHOR_TEXTS = list(SPORT_ANCHORS.values())

# ── Naloži YAML ───────────────────────────────────────────────────────────────
if not os.path.exists(args.input):
    print(f"NAPAKA: {args.input} ne obstaja.")
    sys.exit(1)

size_mb = os.path.getsize(args.input) // (1024 * 1024)
print(f"Berem {args.input} ({size_mb} MB)...")

with open(args.input, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print(f"Skupaj: {len(data)} clankov")

# ── Filtriraj sportne clanke ──────────────────────────────────────────────────
rows = []
for doc in data:
    if str(doc.get("topics", "")).strip() != "sport":
        continue
    lead       = (doc.get("lead") or "").strip()
    title      = (doc.get("title") or "").strip()
    keywords   = ", ".join(doc.get("keywords") or [])
    paragraphs = doc.get("paragraphs") or []
    body       = lead + " " + " ".join(str(p) for p in paragraphs if p)
    rows.append({
        "id":         str(doc.get("id", "")),
        "url":        doc.get("url", "") or "",
        "date":       str(doc.get("date", "") or ""),
        "title":      title,
        "lead":       lead,
        "body":       body.strip(),
        "paragraphs": paragraphs,
        "keywords":   keywords,
        "_embed_text": f"{title} {lead} {keywords}",
    })

df = pd.DataFrame(rows).drop_duplicates("id")
print(f"Sportnih clankov (brez duplikatov): {len(df)}")

# ── Model ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nalagam model... (device: {device})")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# ── Embed sidra ───────────────────────────────────────────────────────────────
print("Embedam sidra...")
anchor_emb = model.encode(ANCHOR_TEXTS, normalize_embeddings=True,
                          show_progress_bar=False)  # (n_sports, dim)

# ── Embed clanki ──────────────────────────────────────────────────────────────
texts = df["_embed_text"].tolist()
print(f"Embedam {len(texts)} clankov (batch={args.batch})...")

article_emb = model.encode(
    texts,
    batch_size=args.batch,
    normalize_embeddings=True,
    show_progress_bar=True,
    device=device,
)  # (n_articles, dim)

# ── Cosine similarity → klasifikacija ─────────────────────────────────────────
# Ker so embeddingi normirani, je dot product = cosine similarity
sim = article_emb @ anchor_emb.T  # (n_articles, n_sports)

# Top-3 indeksi za vsak članek
top3_idx = np.argsort(sim, axis=1)[:, -3:][:, ::-1]  # (n, 3), padajoče

best_idx  = top3_idx[:, 0]
best_conf = sim[np.arange(len(sim)), best_idx]
sec_conf  = sim[np.arange(len(sim)), top3_idx[:, 1]]

sports     = [SPORT_NAMES[i] if c >= args.threshold else "Drugo"
              for i, c in zip(best_idx, best_conf)]
confidence = best_conf.tolist()

df["sport"]       = sports
df["confidence"]  = [round(float(c), 4) for c in confidence]
df["top2_sport"]  = [SPORT_NAMES[i] for i in top3_idx[:, 1]]
df["top2_conf"]   = [round(float(sim[r, i]), 4) for r, i in enumerate(top3_idx[:, 1])]
df["top3_sport"]  = [SPORT_NAMES[i] for i in top3_idx[:, 2]]
df["top3_conf"]   = [round(float(sim[r, i]), 4) for r, i in enumerate(top3_idx[:, 2])]
df["conf_gap"]    = [round(float(a - b), 4) for a, b in zip(best_conf, sec_conf)]
df = df.drop(columns=["_embed_text"])

# ── Shrani ────────────────────────────────────────────────────────────────────
df.to_parquet(args.output, index=False)
print(f"\nShranjeno: {args.output}")

# ── Statistika ────────────────────────────────────────────────────────────────
print("\nPorazdelitev po sportu:")
print(df["sport"].value_counts().to_string())
print(f"\nPovp. confidence: {df['confidence'].mean():.3f}")
print(f"Pod pragom ({args.threshold}) → 'Drugo': {(df['sport'] == 'Drugo').sum()}")
