"""
07_self_training.py

Self-training: nauči klasifikator na ročnih oznakah iz sport_labels.csv,
nato prepiše klasifikacijo v sport_clanki.parquet.

Podpira iterativni self-training: LR pseudo-labeli iz prvega modela
(zanesljivejši od cosine-similarity pseudo-labelov).

Izhod: sport_clanki_trained.parquet (posodobljeno: stolpci sport, confidence)
"""

import argparse
import glob
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

THRESHOLD_DRUGO = 0.35

TARGET_SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis",
]
# "Drugo" ni trening razred — določi se prek praga zaupanja

parser = argparse.ArgumentParser()
parser.add_argument("--labels",          default="sport_labels.csv")
parser.add_argument("--parquet",         default="sport_clanki.parquet",
                    help="Vhodni parquet (originalni, se NE prepiše)")
parser.add_argument("--output",          default="sport_clanki_trained.parquet",
                    help="Izhodni parquet z novo klasifikacijo")
parser.add_argument("--C",               type=float, default=5.0,
                    help="Regularizacijski parameter LogisticRegression (privzeto 5.0)")
parser.add_argument("--threshold",       type=float, default=THRESHOLD_DRUGO,
                    help="Min. zaupanje za dodelitev sporta; pod tem = 'Drugo'")
parser.add_argument("--iter-pseudo-n",   type=int, default=50,
                    help="Top N člankov po sportu za LR pseudo-labele (0 = brez iteracije)")
parser.add_argument("--iter-threshold",  type=float, default=0.95,
                    help="Min. zaupanje LR za pseudo-labele (privzeto 0.95)")
parser.add_argument("--batch",           type=int, default=256)
args = parser.parse_args()

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


def train_and_cv(X, y, C, le):
    clf_cv = LogisticRegression(max_iter=1000, C=C, random_state=42)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf_cv, X, y, cv=cv, scoring="accuracy")
    print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    clf = LogisticRegression(max_iter=1000, C=C, random_state=42)
    clf.fit(X, y)
    return clf


# ── Naloži oznake ─────────────────────────────────────────────────────────────
print(f"Berem {args.labels}...")
labels_df = pd.read_csv(args.labels)
labels_df["id"] = labels_df["id"].astype(str)
n_drugo = (labels_df["sport"] == "Drugo").sum()
labels_df = labels_df[labels_df["sport"] != "Drugo"].copy()
print(f"  Ročnih oznak: {len(labels_df)} (izpuščeno {n_drugo} 'Drugo' → prag zaupanja)")
print(f"  Porazdelitev:\n{labels_df['sport'].value_counts().to_string()}\n")

# ── Naloži parquet ─────────────────────────────────────────────────────────────
print(f"Berem {args.parquet}...")
df = pd.read_parquet(args.parquet)
df["id"] = df["id"].astype(str)
df["_text"] = (
    df["title"].fillna("") + " " +
    df["lead"].fillna("") + " " +
    df["keywords"].fillna("")
)

# ── Embed VSE članke enkrat ───────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nalagam model... (device: {device})")
model_st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

print(f"Embeddam {len(df)} člankov...")
X_all_emb = model_st.encode(
    df["_text"].tolist(),
    batch_size=args.batch,
    normalize_embeddings=True,
    show_progress_bar=True,
    device=device,
)
X_all_kw = keyword_features(df["_text"].tolist())
X_all    = np.hstack([X_all_emb, X_all_kw])
print(f"  Features: {X_all_emb.shape[1]} embedding + {X_all_kw.shape[1]} keyword = {X_all.shape[1]}")

# Indeksi označenih člankov v df
id_to_idx = {id_: i for i, id_ in enumerate(df["id"].tolist())}
manual_ids = labels_df["id"].tolist()
manual_idx = [id_to_idx[id_] for id_ in manual_ids if id_ in id_to_idx]
X_manual   = X_all[manual_idx]
y_manual   = LabelEncoder().fit_transform(labels_df.loc[
    labels_df["id"].isin(df["id"]), "sport"
])

le = LabelEncoder()
le.fit(labels_df["sport"])
y_manual = le.transform(labels_df.loc[labels_df["id"].isin(df["id"]), "sport"])

# ── Korak 1: treniraj na ročnih oznakah ──────────────────────────────────────
print(f"\n── Korak 1: trening na {len(manual_idx)} ročnih oznakah ──")
clf1 = train_and_cv(X_manual, y_manual, args.C, le)

# ── Predikcija vseh (za iterativni self-training) ────────────────────────────
proba1      = clf1.predict_proba(X_all)
pred_idx1   = np.argmax(proba1, axis=1)
pred_sport1 = le.inverse_transform(pred_idx1)
confidence1 = proba1[np.arange(len(proba1)), pred_idx1]

# ── Korak 2 (opcijsko): iterativni self-training z LR pseudo-labeli ──────────
if args.iter_pseudo_n > 0:
    print(f"\n── Korak 2: iterativni self-training "
          f"(top {args.iter_pseudo_n}/šport, prag={args.iter_threshold}) ──")

    labeled_ids = set(labels_df["id"].tolist())
    pseudo_parts = []
    for sport in TARGET_SPORTS:
        # samo neoznačeni članki z visokim zaupanjem
        mask = (
            (pred_sport1 == sport) &
            (confidence1 >= args.iter_threshold) &
            (~df["id"].isin(labeled_ids))
        )
        candidates = df[mask].copy()
        candidates["_conf"] = confidence1[mask]
        top = candidates.nlargest(args.iter_pseudo_n, "_conf")[["id"]].copy()
        top["sport"] = sport
        pseudo_parts.append(top)

    pseudo_df = pd.concat(pseudo_parts, ignore_index=True)
    print(f"  LR pseudo-labelov: {len(pseudo_df)}")
    print(f"  Porazdelitev:\n{pseudo_df['sport'].value_counts().to_string()}\n")

    # Kombiniraj ročne + LR pseudo (ročne imajo prednost)
    combined_df = pd.concat([
        pseudo_df[~pseudo_df["id"].isin(labeled_ids)],
        labels_df[["id", "sport"]],
    ], ignore_index=True)
    print(f"  Skupaj za trening: {len(combined_df)}")

    # Indeksi kombiniranega seta
    comb_ids  = combined_df["id"].tolist()
    comb_idx  = [id_to_idx[id_] for id_ in comb_ids if id_ in id_to_idx]
    X_comb    = X_all[comb_idx]
    y_comb    = le.transform(combined_df.loc[combined_df["id"].isin(df["id"]), "sport"])

    clf_final = train_and_cv(X_comb, y_comb, args.C, le)

    proba_final    = clf_final.predict_proba(X_all)
    pred_idx_final = np.argmax(proba_final, axis=1)
    pred_sport     = le.inverse_transform(pred_idx_final)
    confidence     = proba_final[np.arange(len(proba_final)), pred_idx_final]
else:
    pred_sport = pred_sport1
    confidence = confidence1

# ── Prag zaupanja → Drugo ─────────────────────────────────────────────────────
final_sport = [
    s if c >= args.threshold else "Drugo"
    for s, c in zip(pred_sport, confidence)
]
print(f"\n  Prag zaupanja: {args.threshold}  →  {sum(s == 'Drugo' for s in final_sport)} člankov → 'Drugo'")

df["sport"]      = final_sport
df["confidence"] = np.round(confidence, 4)
df = df.drop(columns=["_text"])

# ── Shrani ────────────────────────────────────────────────────────────────────
df.to_parquet(args.output, index=False)
print(f"\nShranjeno: {args.output}")
print(f"Originalni parquet nespremenjen: {args.parquet}")

print("\nNova porazdelitev:")
print(df["sport"].value_counts().to_string())
print(f"\nPovp. confidence: {df['confidence'].mean():.3f}")

# ── Posodobi sport stolpec v sentiment parquetu ───────────────────────────────
sent_files = glob.glob("sentiment_xlm_roberta_base_sentiment_multilingual*.parquet")
sent_files = [f for f in sent_files if "_paragraphs" not in f]
if sent_files:
    sent_path = sent_files[0]
    print(f"\nPosodabljam sport v {sent_path}...")
    sent = pd.read_parquet(sent_path)
    new_sports = df[["id", "sport"]].copy()
    new_sports["id"] = new_sports["id"].astype(str)
    sent["id"] = sent["id"].astype(str)
    sent = sent.drop(columns=["sport"]).merge(new_sports, on="id", how="left")
    sent["sport"] = sent["sport"].fillna("Drugo")
    sent.to_parquet(sent_path, index=False)
    print(f"  Shranjeno: {sent_path}")
else:
    print("\nOpozorilo: sentiment parquet ni najden, preskoči posodobitev sporta.")
