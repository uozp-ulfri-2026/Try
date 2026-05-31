"""
05_precompute.py

Iz sentiment.parquet producira JSON datoteke za frontend:
  data/sports.json
  data/sport_{id}.json  (za vsakega od 12 sportov)
"""

import json
import os
import re
import unicodedata
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    import lemmagen3
    _lemmatizer = lemmagen3.Lemmatizer("sl")
    def lemmatize(word):
        return _lemmatizer.lemmatize(word)
except ImportError:
    def lemmatize(word):
        return word

INPUT   = "sentiment_xlm_roberta_base_sentiment_multilingual.parquet"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

SPORTS = [
    "Nogomet", "Rokomet", "Alpsko smučanje", "Kolesarstvo", "Košarka",
    "Hokej na ledu", "Atletika", "Smučarski skoki", "Odbojka",
    "Športno plezanje", "Biatlon", "Tenis",
]

SLO_STOPWORDS = [
    # Predlogi, vezniki, členki
    "in", "je", "so", "se", "na", "za", "da", "ki", "v", "z", "s", "ko",
    "pa", "ne", "po", "bi", "ga", "mu", "jo", "jih", "jim", "mi", "me",
    "si", "bo", "bila", "bilo", "bili", "bile", "sem", "smo", "ste",
    "ter", "ali", "kot", "več", "tudi", "le", "že", "še", "od", "do",
    "pri", "med", "nad", "pod", "pred", "ob", "iz", "ta", "to", "te",
    "ti", "tega", "temu", "tem", "tej", "ker", "kar", "kaj", "kdo",
    "kako", "kdaj", "kjer", "če", "toda", "ampak", "a", "o", "ni",
    "niso", "ima", "imajo", "bil", "sta", "bi", "so", "pa",
    # Glagoli brez vsebine
    "biti", "imeti", "reči", "povedati", "priti", "iti", "postati",
    "narediti", "dobiti", "dati", "videti", "znati", "morati", "moči",
    "igrati", "zmagati", "izgubiti", "nastopiti", "tekmovati",
    # Splošni pridevniki / prislovi
    "novo", "novi", "nova", "nove", "nov",
    "prvi", "prva", "prvo", "druge", "drugi", "druga",
    "zelo", "bolj", "manj", "že", "še", "samo", "kar", "prav",
    "veliko", "malo", "vedno", "nikoli", "potem", "nato", "zato",
    # Meseci
    "januar", "februar", "marec", "april", "maj", "junij",
    "julij", "avgust", "september", "oktober", "november", "december",
    # Kratke besede brez smisla
    "an", "en", "ena", "eno", "dva", "tri", "štiri", "pet",
    # Pogosti šum
    "zaradi", "jaz", "on", "zdaj", "svoj", "svojo", "svoje", "svojem",
]

TOP_KEYWORDS_N = 20


def prepare_text_for_tfidf(texts):
    """Lematizira besede in odstranjuje stevilke."""
    result = []
    for text in texts:
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        lemmas = []
        for w in words:
            if re.match(r"^\d+$", w):        # samo stevilke
                continue
            if len(w) <= 2:                  # prekratke besede
                continue
            lemmas.append(lemmatize(w))
        result.append(" ".join(lemmas))
    return result


def normalize_weights(scores):
    """Normalizira uteži na [0.25, 1.0] za vsaj 4× vizualno razliko."""
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return [1.0] * len(scores)
    norm = (scores - scores.min()) / (scores.max() - scores.min())
    return (norm * 0.75 + 0.25).tolist()

TIMELINE_POINTS    = 120
SMOOTHED_POINTS    = 200
LOWESS_K           = 25   # fiksno okno v steviliu clankov (ne dneh)
CONTEXT_DAYS         = 14
PROMINENCE_THRESHOLD = 0.08  # minimum prominence kot delez razpona signala
DATE_START         = pd.Timestamp("2023-04-28")
DATE_END           = pd.Timestamp("2026-03-31")


def to_id(name):
    """'Smučarski skoki' → 'smucarski-skoki'"""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_str.lower()).strip("-")


def compute_lowess(dates, values, n_out):
    """Vrne (dates_out, values_out) z n_out enakomerno razporejenimi tockami.
    Bandwidth = LOWESS_K clankov (fiksno stevilo, ne frakcija dni)."""
    dates = pd.to_datetime(dates)
    x = (dates - dates.min()).days.values.astype(float)
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    n = mask.sum()
    if n < 5:
        return [], []
    frac = min(LOWESS_K / n, 0.5)
    lw = lowess(y[mask], x[mask], frac=frac, it=1)
    x_out = np.linspace(x[mask].min(), x[mask].max(), n_out)
    y_out = np.interp(x_out, lw[:, 0], lw[:, 1])
    dates_out = [dates.min() + timedelta(days=float(d)) for d in x_out]

    # Odreži robove kjer je LOWESS nezanesljiv (enostransko okno)
    # Odreži robove kjer je LOWESS nezanesljiv (enostransko okno)
    trim = max(1, int(n_out * frac / 2))
    dates_out = dates_out[trim:-trim]
    y_out     = y_out[trim:-trim]

    return dates_out, y_out.tolist()


def detect_extremes(smoothed_dates, smoothed_values, df_sport, mean_val, background_texts):
    """Najde lokalne ekstreme na LOWESS krivulji z prominence-based detekcijo."""
    sig = np.array(smoothed_values)
    extremes = []
    ext_id = 0

    signal_range = sig.max() - sig.min()
    prominence   = signal_range * PROMINENCE_THRESHOLD

    peaks,   _ = find_peaks( sig, prominence=prominence)
    valleys, _ = find_peaks(-sig, prominence=prominence)

    # Vrh je veljaven samo ce je nad povprecjem, dolina samo ce je pod
    peaks   = [i for i in peaks   if sig[i] > mean_val]
    valleys = [i for i in valleys if sig[i] < mean_val]
    all_extremes = sorted(set(peaks + valleys))

    for idx in all_extremes:
        val  = sig[idx]
        delta = round(float(val - mean_val), 4)

        etype = "peak" if val > mean_val else "valley"
        edate = smoothed_dates[idx]

        # Kontekstni clanki v oknu ±CONTEXT_DAYS
        d_from = edate - timedelta(days=CONTEXT_DAYS)
        d_to   = edate + timedelta(days=CONTEXT_DAYS)
        ctx = df_sport[
            (df_sport["date"] >= d_from) & (df_sport["date"] <= d_to)
        ]

        if len(ctx) == 0:
            continue

        # Dolina zahteva vsaj en negativen članek v oknu
        if etype == "valley" and not (ctx["sentiment_label"] == "negative").any():
            continue

        context_ids = ctx["id"].astype(str).tolist()

        # Naslov najekstremnejšega clanka kot label
        if etype == "peak":
            label_row = ctx.loc[ctx["sentiment_raw"].idxmax()]
        else:
            label_row = ctx.loc[ctx["sentiment_raw"].idxmin()]
        label = str(label_row.get("title", ""))[:80]

        # TF-IDF kljucne besede — IDF iz celotnega sporta, TF iz konteksta
        ctx_texts = ctx["title"].dropna().tolist()
        keywords = []
        if len(ctx_texts) >= 2:
            try:
                bg_lemmatized  = prepare_text_for_tfidf(background_texts)
                ctx_lemmatized = prepare_text_for_tfidf(ctx_texts)

                # Fit na ozadju (vsi clanki sporta) → IDF penalizira pogoste besede
                vec = TfidfVectorizer(
                    max_features=TOP_KEYWORDS_N * 5,  # vecji vocab, potem izberemo top
                    stop_words=SLO_STOPWORDS,
                    min_df=2,
                    ngram_range=(1, 1),
                )
                vec.fit(bg_lemmatized)

                # Transform samo kontekstne clanke
                mat = vec.transform(ctx_lemmatized)

                # Poudarek na člankih ki se ujemajo s tipom ekstrema
                SENTIMENT_BOOST = 5.0
                match_label = "positive" if etype == "peak" else "negative"
                raw_w = np.array([
                    SENTIMENT_BOOST if lbl == match_label else 1.0
                    for lbl in ctx["sentiment_label"].values
                ])
                w = raw_w / raw_w.sum()
                scores = mat.multiply(w[:, np.newaxis]).sum(axis=0).A1
                words  = vec.get_feature_names_out()

                # Top N po score
                pairs  = sorted(zip(words, scores), key=lambda x: -x[1])[:TOP_KEYWORDS_N]
                pairs  = [(w, s) for w, s in pairs if s > 0]

                weights = normalize_weights([s for _, s in pairs])
                keywords = [
                    {"word": w, "weight": round(float(wt), 4)}
                    for (w, _), wt in zip(pairs, weights)
                ]
            except Exception:
                pass

        ext_id += 1
        extremes.append({
            "id":                  f"e{ext_id}",
            "type":                etype,
            "date":                edate.strftime("%Y-%m-%d"),
            "label":               label,
            "delta":               delta,
            "context_article_ids": context_ids,
            "keywords":            keywords,
        })

    return extremes


# ── Naloži podatke ─────────────────────────────────────────────────────────────
print(f"Berem {INPUT}...")
df = pd.read_parquet(INPUT)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].dt.tz is not None:
    df["date"] = df["date"].dt.tz_localize(None)
df = df.dropna(subset=["date", "sentiment_raw"])
df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)]
df = df[df["sentiment_label"] != "neutral"]

date_range = [DATE_START.strftime("%Y-%m-%d"), DATE_END.strftime("%Y-%m-%d")]
print(f"Obdobje: {date_range[0]} – {date_range[1]}")
print(f"Clankov po filtrih (brez nevtralnih): {len(df)}")

# Skupna casovna os za timeline (120 tock, enaka za vse športe)
timeline_dates = pd.date_range(DATE_START, DATE_END, periods=TIMELINE_POINTS)

sports_summary = []

for sport_name in SPORTS:
    sport_id = to_id(sport_name)
    df_sport = df[df["sport"] == sport_name].copy().sort_values("date")

    if len(df_sport) < 10:
        print(f"  Preskacam {sport_name} (premalo clankov)")
        continue

    print(f"  {sport_name} ({len(df_sport)} clankov)...")

    # Dnevno povprecje
    daily = df_sport.groupby(df_sport["date"].dt.date)["sentiment_raw"].mean()
    daily.index = pd.to_datetime(daily.index)

    # LOWESS za smoothed (~200 tock z datumi)
    sm_dates, sm_values = compute_lowess(daily.index, daily.values, SMOOTHED_POINTS)
    if not sm_dates:
        continue

    mean_val = float(np.mean(sm_values))

    # Timeline: 120 tock na skupni osi, interpolirano iz LOWESS
    lw_x = np.array([(d - sm_dates[0]).days for d in sm_dates], dtype=float)
    tl_x = np.array([(d - sm_dates[0]).days for d in timeline_dates
                     if sm_dates[0] <= d <= sm_dates[-1]], dtype=float)
    timeline_values = np.interp(tl_x, lw_x, sm_values).tolist()
    # Dopolni na tocno 120 vrednosti (ce je sport krajsi od skupne osi)
    if len(timeline_values) < TIMELINE_POINTS:
        timeline_values = [None] * (TIMELINE_POINTS - len(timeline_values)) + timeline_values

    # Boxplot iz raw sentiment_raw vrednosti
    s = df_sport["sentiment_raw"].values
    boxplot = {
        "min":    round(float(np.percentile(s, 0)),  4),
        "q1":     round(float(np.percentile(s, 25)), 4),
        "median": round(float(np.percentile(s, 50)), 4),
        "q3":     round(float(np.percentile(s, 75)), 4),
        "max":    round(float(np.percentile(s, 100)), 4),
    }

    # Ekstremi
    bg_texts = df_sport["title"].dropna().tolist()
    extremes = detect_extremes(sm_dates, sm_values, df_sport, mean_val, bg_texts)

    # Articles za detail stran
    articles = [
        {
            "id":        str(row["id"]),
            "date":      row["date"].strftime("%Y-%m-%d"),
            "title":     str(row.get("title", "") or ""),
            "url":       str(row.get("url", "") or ""),
            "sentiment": round(float(row["sentiment_raw"]), 4),
        }
        for _, row in df_sport.iterrows()
    ]

    # Smoothed za detail stran
    smoothed = [
        {"date": d.strftime("%Y-%m-%d"), "value": round(v, 4)}
        for d, v in zip(sm_dates, sm_values)
    ]

    # Shrani sport_{id}.json
    sport_detail = {
        "id":         sport_id,
        "name":       sport_name,
        "n_articles": len(df_sport),
        "date_range": date_range,
        "articles":   articles,
        "smoothed":   smoothed,
        "extremes":   extremes,
    }
    out_path = os.path.join(OUT_DIR, f"sport_{sport_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sport_detail, f, ensure_ascii=False)
    print(f"    → {out_path}  ({len(extremes)} ekstremov)")

    # Dodaj v sports summary
    sports_summary.append({
        "id":         sport_id,
        "name":       sport_name,
        "n_articles": len(df_sport),
        "timeline":   [round(v, 4) if v is not None else None
                       for v in timeline_values],
        "boxplot":    boxplot,
    })

# Shrani sports.json
sports_json = {
    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    "date_range":   date_range,
    "sports":       sports_summary,
}
sports_path = os.path.join(OUT_DIR, "sports.json")
with open(sports_path, "w", encoding="utf-8") as f:
    json.dump(sports_json, f, ensure_ascii=False)

print(f"\nShranjeno: {sports_path}")
print(f"Sportov:   {len(sports_summary)}")
