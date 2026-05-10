import pandas as pd
import numpy as np
import json
import sys
import ruptures as rpt
from scipy.signal import argrelextrema
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.nonparametric.smoothers_lowess import lowess

SLO_STOPWORDS = [
    "in", "je", "so", "se", "na", "za", "da", "ki", "v", "z", "s", "ko",
    "pa", "ne", "po", "bi", "ga", "mu", "jo", "jih", "jim", "mi", "me",
    "si", "bo", "bila", "bilo", "bili", "bile", "sem", "smo", "ste",
    "ter", "ali", "kot", "več", "tudi", "le", "že", "še", "od", "do",
    "pri", "med", "nad", "pod", "pred", "ob", "iz", "ta", "to", "te",
    "ti", "tega", "temu", "tem", "tej", "ker", "kar", "kaj", "kdo",
    "kako", "kdaj", "kjer", "če", "toda", "ampak", "a", "o", "ni",
    "niso", "ima", "imajo", "bil", "sta", "so", "je", "pa", "ne",
]

def process_sport(csv_file, sport_name):
    print(f"\nObdelujem {sport_name}...")
    sentiment = pd.read_csv(csv_file, parse_dates=["date"])
    sentiment = sentiment.dropna(subset=["date", "sentiment_raw"]).sort_values("date")

    # Uteženo dnevno povprečje
    def weighted_mean(group):
        weights = group["sentiment_raw"].abs() + 0.01
        return (group["sentiment_raw"] * weights).sum() / weights.sum()

    daily = sentiment.groupby(sentiment["date"].dt.date).apply(
        weighted_mean, include_groups=False
    ).reset_index()
    daily.columns = ["date", "sentiment"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["date"] >= "2023-05-01"].sort_values("date").reset_index(drop=True)

    # Ratio metrika: % pozitivnih - % negativnih
    def pos_ratio(group):
        n = len(group)
        if n == 0:
            return 0
        n_pos = (group["sentiment_label"] == "positive").sum()
        n_neg = (group["sentiment_label"] == "negative").sum()
        return (n_pos - n_neg) / n

    daily_ratio = sentiment.groupby(sentiment["date"].dt.date).apply(
        pos_ratio, include_groups=False
    ).reset_index()
    daily_ratio.columns = ["date", "ratio"]
    daily_ratio["date"] = pd.to_datetime(daily_ratio["date"])
    daily = daily.merge(daily_ratio, on="date", how="left")

    # LOWESS tudi za ratio
    daily_full_r = daily.set_index("date").reindex(
        pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    ).rename_axis("date").reset_index()
    xr = (daily_full_r["date"] - daily_full_r["date"].min()).dt.days.values
    yr = daily_full_r["ratio"].values
    mask_r = ~np.isnan(yr)
    lowess_r = lowess(yr[mask_r], xr[mask_r], frac=0.08, it=1)
    daily["ratio_smoothed"] = np.interp(
        (daily["date"] - daily_full_r["date"].min()).dt.days.values,
        lowess_r[:, 0], lowess_r[:, 1]
    )

    # LOWESS
    daily_full = daily.set_index("date").reindex(
        pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    ).rename_axis("date").reset_index()
    x = (daily_full["date"] - daily_full["date"].min()).dt.days.values
    y = daily_full["sentiment"].values
    mask_valid = ~np.isnan(y)
    lowess_result = lowess(y[mask_valid], x[mask_valid], frac=0.08, it=1)
    daily["smoothed"] = np.interp(
        (daily["date"] - daily_full["date"].min()).dt.days.values,
        lowess_result[:, 0], lowess_result[:, 1]
    )

    # Changepoint detection — lokalni ekstremi na LOWESS krivulji
    signal = daily["smoothed"].ffill().fillna(0).values

    # Najdi lokalne min in max z različnimi order-ji, izberi order ki da 8-12 točk
    best_order = 20
    for order in [10, 15, 20, 25, 30, 40, 50]:
        maxima = argrelextrema(signal, np.greater_equal, order=order)[0]
        minima = argrelextrema(signal, np.less_equal, order=order)[0]
        all_extrema = sorted(set(maxima.tolist() + minima.tolist()))
        # Odstrani preblizu skupaj (min 14 dni razmika)
        filtered = []
        for idx in all_extrema:
            if not filtered or idx - filtered[-1] >= 14:
                filtered.append(idx)
        if 6 <= len(filtered) <= 14:
            best_order = order
            break

    maxima = argrelextrema(signal, np.greater_equal, order=best_order)[0]
    minima = argrelextrema(signal, np.less_equal, order=best_order)[0]
    all_extrema = sorted(set(maxima.tolist() + minima.tolist()))
    breakpoints = []
    for idx in all_extrema:
        if not breakpoints or idx - breakpoints[-1] >= 14:
            breakpoints.append(idx)

    print(f"  Changepointov: {len(breakpoints)}, order={best_order}")

    # Povprečni smoothed sentiment za threshold
    mean_smoothed = daily["smoothed"].mean()

    # Segmenti
    segments = []
    prev = 0
    for bp in breakpoints:
        segments.append((prev, bp))
        prev = bp
    segments.append((prev, len(daily)))

    result_segments = []
    for i, (start, end) in enumerate(segments):
        if end - start < 3:
            continue
        date_start = daily.iloc[start]["date"]
        date_end = daily.iloc[min(end, len(daily)-1)]["date"]
        cp_date = daily.iloc[start]["date"] if start > 0 else None

        # Članki v segmentu
        mask = (sentiment["date"] >= date_start) & (sentiment["date"] < date_end)
        seg_df = sentiment[mask].copy()

        if len(seg_df) < 3:
            continue

        avg_sentiment = float(seg_df["sentiment_raw"].mean())
        smoothed_at_cp = float(daily.iloc[start]["smoothed"]) if start < len(daily) else avg_sentiment
        above_mean = smoothed_at_cp > mean_smoothed

        # Top naslovi
        title_col = "title" if "title" in seg_df.columns else "lead"
        top_articles = seg_df.nlargest(5, "sentiment_raw")[title_col].dropna().tolist()

        # TF-IDF
        texts = seg_df["lead"].dropna().tolist()
        tfidf_words = []
        if len(texts) >= 3:
            vec = TfidfVectorizer(max_features=8, stop_words=SLO_STOPWORDS, min_df=2)
            try:
                tfidf_matrix = vec.fit_transform(texts)
                scores = tfidf_matrix.mean(axis=0).A1
                words = vec.get_feature_names_out()
                tfidf_words = [
                    {"word": w, "score": round(float(s), 4)}
                    for w, s in sorted(zip(words, scores), key=lambda x: -x[1])
                ]
            except:
                pass

        result_segments.append({
            "segment_id": i,
            "date_start": str(date_start.date()),
            "date_end": str(date_end.date()),
            "cp_date": str(cp_date.date()) if cp_date is not None else None,
            "avg_sentiment": round(avg_sentiment, 4),
            "smoothed_at_cp": round(smoothed_at_cp, 4),
            "above_mean": bool(above_mean),
            "n_articles": int(len(seg_df)),
            "top_articles": top_articles,
            "tfidf": tfidf_words,
        })

    # Timeline za graf
    timeline = [
        {
            "date": str(row["date"].date()),
            "sentiment": round(float(row["sentiment"]), 4),
            "smoothed": round(float(row["smoothed"]), 4),
            "ratio": round(float(row["ratio"]), 4) if "ratio" in row and not pd.isna(row["ratio"]) else 0,
            "ratio_smoothed": round(float(row["ratio_smoothed"]), 4) if "ratio_smoothed" in row and not pd.isna(row["ratio_smoothed"]) else 0,
        }
        for _, row in daily.iterrows()
    ]

    # Asimetrija
    above = [s for s in result_segments if s["above_mean"]]
    below = [s for s in result_segments if not s["above_mean"]]
    asym = {
        "above_mean": round(np.mean([s["avg_sentiment"] for s in above]), 4) if above else 0,
        "below_mean": round(np.mean([s["avg_sentiment"] for s in below]), 4) if below else 0,
        "n_above": len(above),
        "n_below": len(below),
    }
    asym["difference"] = round(asym["above_mean"] - asym["below_mean"], 4)

    return {
        "sport": sport_name,
        "timeline": timeline,
        "segments": result_segments,
        "asymmetry": asym,
        "mean_smoothed": round(float(mean_smoothed), 4),
    }

# Procesiraj vse športe
sports = [
    ("skoki_sentiment.csv", "Smučarski skoki"),
    ("kolesarstvo_sentiment.csv", "Kolesarstvo"),
    ("kosarka_sentiment.csv", "Košarka"),
    ("hokej_sentiment.csv", "Hokej"),
    ("alpsko_smucanje_sentiment.csv", "Alpsko smučanje"),
    ("biatlon_sentiment.csv", "Biatlon"),
]

output = {}
for csv_file, name in sports:
    try:
        output[name] = process_sport(csv_file, name)
        print(f"  Asimetrija: {output[name]['asymmetry']}")
    except Exception as e:
        print(f"  NAPAKA: {e}")

with open("segments.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\nShranjeno: segments.json")
