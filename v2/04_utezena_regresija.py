"""
04_utezena_regresija.py

Nauci optimalne utezi za kombinacijo lead/first/middle/last sentimenta
z Ridge regresijo in 5-fold cross-validation.

Izhod: utezi.json  (utezi za vsak tip odstavka)
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

LABELS_CSV      = "/home/miha/Downloads/oznake.csv"
PARA_PARQUET    = "sentiment_xlm_roberta_base_sentiment_multilingual_paragraphs.parquet"
OUTPUT_WEIGHTS  = "utezi.json"

LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# ── Naloži podatke ─────────────────────────────────────────────────────────────
labels = pd.read_csv(LABELS_CSV)
labels["id"] = labels["id"].astype(str)
labels = labels[["id", "label"]]
labels["y"] = labels["label"].map(LABEL_MAP)
labels = labels.dropna(subset=["y"])

para_df = pd.read_parquet(PARA_PARQUET)
para_df["article_id"] = para_df["article_id"].astype(str)

# ── Izracunaj features po tipu odstavka ───────────────────────────────────────
# Za vsak clanek: mean sentiment po tipu (lead, first, middle, last)
pivot = para_df.groupby(["article_id", "para_type"])["sentiment_raw"].mean().unstack()

# Ce tipa ni, vzami mean vseh odstavkov clanka kot fallback
article_mean = para_df.groupby("article_id")["sentiment_raw"].mean()
for col in ["lead", "first", "middle", "last"]:
    if col not in pivot.columns:
        pivot[col] = np.nan
    pivot[col] = pivot[col].fillna(article_mean)

pivot = pivot.reset_index().rename(columns={"article_id": "id"})

# ── Spoji z oznakami ───────────────────────────────────────────────────────────
df = labels.merge(pivot, on="id", how="inner")
print(f"Vzorcev za regresijo: {len(df)}")
print(f"Distribucija: {labels['label'].value_counts().to_dict()}")

features = ["lead", "first", "middle", "last"]
X = df[features].values
y = df["y"].values

# ── Baseline: navadni povprecek (enake utezi) ──────────────────────────────────
def score_kappa(y_true, y_pred_cont):
    pred_labels = np.where(y_pred_cont > 0.33, "positive",
                  np.where(y_pred_cont < -0.33, "negative", "neutral"))
    true_labels = np.where(y_true == 1.0, "positive",
                  np.where(y_true == -1.0, "negative", "neutral"))
    return cohen_kappa_score(true_labels, pred_labels)

baseline_pred = X.mean(axis=1)
print(f"\nBaseline κ (enake utezi): {score_kappa(y, baseline_pred):.3f}")

# ── Ridge regresija s 5-fold CV ────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# RidgeCV sam poisce najboljso vrednost alpha
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
model = RidgeCV(alphas=alphas, cv=kf, fit_intercept=True)
model.fit(X, y)

print(f"Izbrana alpha (regularizacija): {model.alpha_:.3f}")

# Kappa po foldih
kappas = []
for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    m = RidgeCV(alphas=alphas, fit_intercept=True)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    kappas.append(score_kappa(y_te, pred))

print(f"\n5-fold CV κ: {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
print(f"Po foldih:   {[f'{k:.3f}' for k in kappas]}")

# ── Utezi finalnega modela ─────────────────────────────────────────────────────
print(f"\nNaucene utezi:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:10s}: {coef:+.4f}")
print(f"  intercept : {model.intercept_:+.4f}")

# ── Shrani utezi ──────────────────────────────────────────────────────────────
weights = {feat: float(coef) for feat, coef in zip(features, model.coef_)}
weights["intercept"] = float(model.intercept_)
weights["alpha"] = float(model.alpha_)

with open(OUTPUT_WEIGHTS, "w") as f:
    json.dump(weights, f, indent=2)

print(f"\nUtezi shranjene: {OUTPUT_WEIGHTS}")
