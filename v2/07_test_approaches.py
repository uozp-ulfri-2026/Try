"""
07_test_approaches.py

Primerja pristope klasifikacije sportov na rocnih oznakah.
Samo CV accuracy — ne re-klasificira vseh clankov, ne pishe v parquet.
"""

import argparse
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--labels",   default="sport_labels.csv")
parser.add_argument("--parquet",  default="sport_clanki.parquet",
                    help="Originalni parquet (za pseudo-labele)")
parser.add_argument("--pseudo-n", type=int, default=150)
parser.add_argument("--batch",    type=int, default=256)
args = parser.parse_args()

TARGET_SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis", "Drugo",
]

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
    sports = [s for s in TARGET_SPORTS if s in SPORT_KEYWORDS]
    feats  = np.zeros((len(texts), len(sports)), dtype=np.float32)
    for j, sport in enumerate(sports):
        pattern = "|".join(re.escape(kw) for kw in SPORT_KEYWORDS[sport])
        for i, text in enumerate(texts):
            if re.search(pattern, text.lower()):
                feats[i, j] = 1.0
    return feats


def cv_score(clf, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def make_text(row):
    return (row["title"] or "") + " " + (row["lead"] or "") + " " + (row["keywords"] or "")


# ── Naloži ────────────────────────────────────────────────────────────────────
print(f"Berem {args.labels}...")
labels_df = pd.read_csv(args.labels)
labels_df["id"] = labels_df["id"].astype(str)
print(f"  Ročnih oznak: {len(labels_df)}")
print(f"  Porazdelitev:\n{labels_df['sport'].value_counts().to_string()}\n")

print(f"Berem {args.parquet}...")
df = pd.read_parquet(args.parquet)
df["id"] = df["id"].astype(str)

# ── Sestavi pseudo-labele (enako kot v 07) ────────────────────────────────────
manual_counts  = labels_df["sport"].value_counts().to_dict()
sorted_by_conf = df[df["sport"].isin(TARGET_SPORTS)].sort_values("confidence", ascending=False)
pseudo_parts   = []
for sport in TARGET_SPORTS:
    n_manual = manual_counts.get(sport, 0)
    n_pseudo = max(0, args.pseudo_n - n_manual)
    if n_pseudo == 0:
        continue
    chunk = sorted_by_conf[sorted_by_conf["sport"] == sport].head(n_pseudo)[["id", "sport"]]
    pseudo_parts.append(chunk)
pseudo   = pd.concat(pseudo_parts, ignore_index=True) if pseudo_parts else pd.DataFrame(columns=["id", "sport"])
combined = pseudo[~pseudo["id"].isin(labels_df["id"])].copy()
combined = pd.concat([combined, labels_df[["id", "sport"]]], ignore_index=True)
print(f"Pseudo-labelov: {len(pseudo)}  |  Skupaj: {len(combined)}\n")

# ── Embed ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nalagam model... (device: {device})")
model_st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# A: samo ročne oznake
lm = df[df["id"].isin(labels_df["id"])].copy()
lm = lm.merge(labels_df[["id", "sport"]], on="id", suffixes=("_orig", ""))
lm["_text"] = lm.apply(make_text, axis=1)

print(f"Embeddam {len(lm)} ročnih oznak...")
Xm_emb = model_st.encode(lm["_text"].tolist(), batch_size=args.batch,
                          normalize_embeddings=True, show_progress_bar=True, device=device)
Xm_kw  = keyword_features(lm["_text"].tolist())
Xm     = np.hstack([Xm_emb, Xm_kw])
le     = LabelEncoder()
ym     = le.fit_transform(lm["sport"])

# B: ročne + pseudo
lc = df[df["id"].isin(combined["id"])].copy()
lc = lc.merge(combined[["id", "sport"]], on="id", suffixes=("_orig", ""))
lc["_text"] = lc.apply(make_text, axis=1)

print(f"Embeddam {len(lc)} (ročne + pseudo)...")
Xc_emb = model_st.encode(lc["_text"].tolist(), batch_size=args.batch,
                          normalize_embeddings=True, show_progress_bar=True, device=device)
Xc_kw  = keyword_features(lc["_text"].tolist())
Xc     = np.hstack([Xc_emb, Xc_kw])
yc     = LabelEncoder().fit_transform(lc["sport"])

# ── Primerjava ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  {'Pristop':<35} │ CV acc  │    N")
print("=" * 60)

results = [
    ("LR (C=1), samo ročne",
     LogisticRegression(max_iter=1000, C=1.0, random_state=42), Xm, ym),
    ("LR (C=5), samo ročne",
     LogisticRegression(max_iter=1000, C=5.0, random_state=42), Xm, ym),
    (f"LR (C=1), ročne + pseudo (n={args.pseudo_n})",
     LogisticRegression(max_iter=1000, C=1.0, random_state=42), Xc, yc),
    (f"LR (C=1), samo embedding (brez KW)",
     LogisticRegression(max_iter=1000, C=1.0, random_state=42), Xm_emb, ym),
    ("k-NN (k=5), samo ročne",
     KNeighborsClassifier(n_neighbors=5,  metric="cosine"), Xm, ym),
    ("k-NN (k=10), samo ročne",
     KNeighborsClassifier(n_neighbors=10, metric="cosine"), Xm, ym),
    ("k-NN (k=5), samo embedding (brez KW)",
     KNeighborsClassifier(n_neighbors=5,  metric="cosine"), Xm_emb, ym),
]

for name, clf, X, y in results:
    mu, sd = cv_score(clf, X, y)
    print(f"  {name:<35} │  {mu:.3f}   │ {len(y):4d}")

# Dodaten test: LR C=5, brez Drugo razreda, prag zaupanja
print(f"\n  --- prag zaupanja (Drugo = fallback, ne razred) ---")
THRESHOLD = 0.40
le_full   = LabelEncoder().fit(lm["sport"])
idx_drugo = np.where(le_full.classes_ == "Drugo")[0]

mask_train = ym != (idx_drugo[0] if len(idx_drugo) else -1)
X12 = Xm[mask_train]
y12 = ym[mask_train]
le12 = LabelEncoder()
sport12 = lm["sport"].values[mask_train]
y12_enc = le12.fit_transform(sport12)

cv12 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for threshold in [0.35, 0.40, 0.45]:
    y_true_all2, y_pred_all2 = [], []
    for train_idx, val_idx in cv12.split(X12, y12_enc):
        clf12 = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
        clf12.fit(X12[train_idx], y12_enc[train_idx])
        proba12 = clf12.predict_proba(X12[val_idx])
        conf12  = proba12.max(axis=1)
        pred12  = le12.inverse_transform(proba12.argmax(axis=1))
        pred12  = np.where(conf12 >= threshold, pred12, "Drugo")
        true12  = sport12[val_idx]
        y_true_all2.extend(true12)
        y_pred_all2.extend(pred12)
    acc = np.mean(np.array(y_true_all2) == np.array(y_pred_all2))
    n_drugo_pred = sum(p == "Drugo" for p in y_pred_all2)
    print(f"  LR C=5, 12 razredov, prag={threshold}       │  {acc:.3f}   │  (Drugo: {n_drugo_pred}/{len(y_pred_all2)})")

print("=" * 60)

# Per-sport breakdown za najboljši pristop (LR samo ročne)
print("\nPer-sport accuracy (LR C=1, samo ročne, 5-fold):")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
y_true_all, y_pred_all = [], []
for train_idx, val_idx in cv.split(Xm, ym):
    clf.fit(Xm[train_idx], ym[train_idx])
    y_true_all.extend(ym[val_idx])
    y_pred_all.extend(clf.predict(Xm[val_idx]))

print(classification_report(
    y_true_all, y_pred_all,
    target_names=le.classes_,
    digits=2,
    zero_division=0,
))
