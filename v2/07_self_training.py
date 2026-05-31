"""
07_self_training.py

Self-training: nauči klasifikator na ročnih oznakah iz sport_labels.csv,
nato prepiše klasifikacijo v sport_clanki.parquet.

Izhod: sport_clanki.parquet (posodobljeno: stolpci sport, confidence)
"""

import argparse
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--labels",       default="sport_labels.csv")
parser.add_argument("--parquet",      default="sport_clanki.parquet",
                    help="Vhodni parquet (originalni, se NE prepiše)")
parser.add_argument("--output",       default="sport_clanki_trained.parquet",
                    help="Izhodni parquet z novo klasifikacijo")
parser.add_argument("--pseudo-n",     type=int, default=150,
                    help="Top N člankov po sportu za pseudo-labele (privzeto 150)")
parser.add_argument("--batch",        type=int, default=256)
args = parser.parse_args()

TARGET_SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis", "Drugo",
]

# Ključne besede za vsak šport — prisotnost v besedilu je močan signal
SPORT_KEYWORDS = {
    "Nogomet":          ["nogomet", "nogometaš", "nogometaši", "futsal"],
    "Rokomet":          ["rokomet", "rokometaš", "rokometaši", "rokometni", "rokometna"],
    "Alpsko smučanje":  ["alpsko smučanje", "alpski smučar", "slalom", "veleslalom", "smuk", "superveleslalom"],
    "Kolesarstvo":      ["kolesarstvo", "kolesar", "kolesarji", "kolesarski"],
    "Košarka":          ["košarka", "košarkar", "košarkarji", "košarkarski", "nba", "euroliga"],
    "Hokej na ledu":    ["hokej", "hokejist", "hokejisti", "hokejaš", "hokejski"],
    "Atletika":         ["atletika", "atlet", "atleti", "atletski", "atletinja"],
    "Smučarski skoki":  ["smučarski skoki", "smučarski skok", "skakalec", "skakalci", "skakalnica"],
    "Odbojka":          ["odbojka", "odbojkar", "odbojkarji", "odbojkarski"],
    "Športno plezanje": ["plezanje", "plezalec", "plezalci", "plezalni", "balvansko", "težavnostno"],
    "Biatlon":          ["biatlon", "biatlonec", "biatlonci", "biatlonka"],
    "Tenis":            ["tenis", "teniški", "tenisač", "tenisači", "atp", "wta"],
}


def keyword_features(texts):
    """Vrne matriko (n_articles, n_sports) z binarnimi keyword značilkami."""
    sports = [s for s in TARGET_SPORTS if s in SPORT_KEYWORDS]
    feats  = np.zeros((len(texts), len(sports)), dtype=np.float32)
    for j, sport in enumerate(sports):
        pattern = "|".join(re.escape(kw) for kw in SPORT_KEYWORDS[sport])
        for i, text in enumerate(texts):
            if re.search(pattern, text.lower()):
                feats[i, j] = 1.0
    return feats

# ── Naloži oznake ─────────────────────────────────────────────────────────────
print(f"Berem {args.labels}...")
labels_df = pd.read_csv(args.labels)
labels_df["id"] = labels_df["id"].astype(str)
print(f"  Ročnih oznak: {len(labels_df)}")
print(f"  Porazdelitev:\n{labels_df['sport'].value_counts().to_string()}\n")

# ── Naloži parquet ─────────────────────────────────────────────────────────────
print(f"Berem {args.parquet}...")
df = pd.read_parquet(args.parquet)
df["id"] = df["id"].astype(str)

# ── Sestavi training set: ročne oznake + pseudo-labeli ────────────────────────
# Pseudo-labeli: top N po confidence za vsak šport — garantira vse športe
pseudo = (
    df[df["sport"].isin(TARGET_SPORTS)]
    .sort_values("confidence", ascending=False)
    .groupby("sport")
    .head(args.pseudo_n)
)[["id", "sport"]].copy()
print(f"  Pseudo-labelov (top {args.pseudo_n}/šport): {len(pseudo)}")
print(f"  Porazdelitev pseudo:\n{pseudo['sport'].value_counts().to_string()}\n")

# Ročne oznake prepišejo pseudo-labele za iste članke
combined = pseudo[~pseudo["id"].isin(labels_df["id"])].copy()
combined = pd.concat([combined, labels_df[["id", "sport"]]], ignore_index=True)
print(f"  Skupaj za trening: {len(combined)}")
print(f"  Porazdelitev:\n{combined['sport'].value_counts().to_string()}\n")

labeled = df[df["id"].isin(combined["id"])].copy()
labeled = labeled.merge(combined[["id", "sport"]], on="id", suffixes=("_orig", ""))

labeled["_text"] = (
    labeled["title"].fillna("") + " " +
    labeled["lead"].fillna("") + " " +
    labeled["keywords"].fillna("")
)

# ── Embeddingi ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nalagam model... (device: {device})")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

train_texts = labeled["_text"].tolist()
print(f"Embedam {len(labeled)} označenih člankov...")
X_emb = model.encode(
    train_texts,
    batch_size=args.batch,
    normalize_embeddings=True,
    show_progress_bar=True,
    device=device,
)
X_kw = keyword_features(train_texts)
X    = np.hstack([X_emb, X_kw])
print(f"  Features: {X_emb.shape[1]} embedding + {X_kw.shape[1]} keyword = {X.shape[1]}")

le = LabelEncoder()
y  = le.fit_transform(labeled["sport"])

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\n5-fold CV (LogisticRegression)...")
clf_cv = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf_cv, X, y, cv=cv, scoring="accuracy")
print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# ── Treniraj na vseh oznakah ──────────────────────────────────────────────────
print("Treniram na vseh oznakah...")
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X, y)

# ── Embedaj VSE članke ────────────────────────────────────────────────────────
df["_text"] = (
    df["title"].fillna("") + " " +
    df["lead"].fillna("") + " " +
    df["keywords"].fillna("")
)

all_texts = df["_text"].tolist()
print(f"\nEmbeddam {len(df)} člankov za re-klasifikacijo...")
X_all_emb = model.encode(
    all_texts,
    batch_size=args.batch,
    normalize_embeddings=True,
    show_progress_bar=True,
    device=device,
)
X_all_kw = keyword_features(all_texts)
X_all    = np.hstack([X_all_emb, X_all_kw])

# ── Predikciaj ────────────────────────────────────────────────────────────────
proba      = clf.predict_proba(X_all)
pred_idx   = np.argmax(proba, axis=1)
pred_sport = le.inverse_transform(pred_idx)
confidence = proba[np.arange(len(proba)), pred_idx]

df["sport"]      = pred_sport
df["confidence"] = np.round(confidence, 4)
df = df.drop(columns=["_text"])

# ── Shrani ────────────────────────────────────────────────────────────────────
df.to_parquet(args.output, index=False)
print(f"\nShranjeno: {args.output}")
print(f"Originalni parquet nespremenjen: {args.parquet}")

print("\nNova porazdelitev:")
print(df["sport"].value_counts().to_string())
print(f"\nPovp. confidence: {df['confidence'].mean():.3f}")
