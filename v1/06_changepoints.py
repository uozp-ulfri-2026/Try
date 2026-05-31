import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ruptures as rpt
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# --- 1. Naloži podatke ---
import sys
INPUT = sys.argv[1] if len(sys.argv) > 1 else "skoki_sentiment.csv"
OUTPUT_PNG = sys.argv[2] if len(sys.argv) > 2 else INPUT.replace("_sentiment.csv", "_changepoints.png").replace(".csv", "_changepoints.png")
sentiment = pd.read_csv(INPUT, parse_dates=["date"])
sentiment = sentiment.dropna(subset=["date", "sentiment_raw"]).sort_values("date")

# Uteženo dnevno povprečje — višji abs(sentiment) = večja teža (vse članke obdržimo)
def weighted_mean(group):
    weights = group["sentiment_raw"].abs() + 0.01  # +0.01 da nevtralni niso popolnoma ignorirani
    return (group["sentiment_raw"] * weights).sum() / weights.sum()

daily = sentiment.groupby(sentiment["date"].dt.date).apply(
    weighted_mean, include_groups=False
).reset_index()
daily.columns = ["date", "sentiment"]
daily["date"] = pd.to_datetime(daily["date"])
daily = daily[daily["date"] >= "2023-05-01"].sort_values("date").reset_index(drop=True)

# LOWESS po dejanskem času
from statsmodels.nonparametric.smoothers_lowess import lowess
daily_full = daily.set_index("date").reindex(
    pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
).rename_axis("date").reset_index()
x = (daily_full["date"] - daily_full["date"].min()).dt.days.values
y = daily_full["sentiment"].values
mask_valid = ~np.isnan(y)
lowess_result = lowess(y[mask_valid], x[mask_valid], frac=0.08, it=1)
from numpy import interp
daily["rolling"] = interp(
    (daily["date"] - daily_full["date"].min()).dt.days.values,
    lowess_result[:, 0], lowess_result[:, 1]
)

# --- 2. Changepoint detection ---
signal = daily["rolling"].ffill().fillna(0).values

model = rpt.Pelt(model="rbf").fit(signal)

# Avtomatsko poišči pen ki da 8-12 changepointov
best_pen = 3
for pen in [1, 2, 3, 4, 5, 6, 8, 10, 15]:
    bps = [bp for bp in model.predict(pen=pen) if bp < len(daily)]
    print(f"pen={pen}: {len(bps)} changepointov")
    if 8 <= len(bps) <= 12:
        best_pen = pen
        break

print(f"\nUporabljam pen={best_pen}")
breakpoints = [bp for bp in model.predict(pen=best_pen) if bp < len(daily)]

print(f"Najdenih {len(breakpoints)} changepointov")
for bp in breakpoints:
    if bp < len(daily):
        print(f"  {daily.iloc[bp]['date'].date()} — sentiment: {daily.iloc[bp]['rolling']:.3f}")

# --- 3. Vizualizacija ---
fig, ax = plt.subplots(figsize=(16, 6))

ax.scatter(daily["date"], daily["sentiment"], alpha=0.2, s=12, color="#bbbbbb", zorder=1)
ax.plot(daily["date"], daily["rolling"], color="#2c3e50", linewidth=2, zorder=2, label="LOWESS (uteženo povprečje)")
ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)

# Označi changepointse
for bp in breakpoints:
    if bp < len(daily):
        d = daily.iloc[bp]["date"]
        ax.axvline(x=d, color="#e74c3c", alpha=0.6, linewidth=1.5, linestyle="--")
        ax.text(d, 0.82, d.strftime("%b %y"), rotation=90, fontsize=7,
                color="#e74c3c", va="top", ha="right")

sport_name = INPUT.replace("_sentiment.csv","").replace(".csv","").replace("_"," ").title()
ax.set_title(f"Sentiment člankov — {sport_name} (RTV MMC 2023–2026)\nRdeče črte = zaznane strukturne spremembe sentimenta", fontsize=13)
ax.set_xlabel("Datum")
ax.set_ylabel("Sentiment (-1 = negativno, +1 = pozitivno)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.set_ylim(-1, 1)
ax.legend()

plt.tight_layout()
plt.savefig("changepoints.png", dpi=150, bbox_inches="tight")
print("\nShranjeno: changepoints.png")

# --- 4. TF-IDF za vsak changepoint ---
# Dodaj slovensko stop besedo listo
SLO_STOPWORDS = [
    "in", "je", "so", "se", "na", "za", "da", "ki", "v", "z", "s", "ko",
    "pa", "ne", "po", "bi", "ga", "mu", "jo", "jih", "jim", "mi", "me",
    "si", "bo", "bila", "bilo", "bili", "bile", "sem", "smo", "ste",
    "ter", "ali", "kot", "več", "tudi", "le", "že", "še", "od", "do",
    "pri", "med", "nad", "pod", "pred", "po", "ob", "iz", "ta", "to",
    "te", "ti", "tega", "temu", "tem", "tej", "ta", "ker", "kar", "kaj",
    "kdo", "kako", "kdaj", "kjer", "ki", "če", "ko", "toda", "ampak",
    "a", "o", "e", "i", "u", "ni", "niso", "ima", "imajo", "ima",
    "da", "ki", "so", "je", "in", "na", "za", "bi", "pa", "ne",
]

print("\n--- TF-IDF top besede po changepointu ---")

# Razdelek med changepointsi
segments = []
prev = 0
for bp in breakpoints:
    if bp < len(daily):
        segments.append((prev, bp))
        prev = bp
segments.append((prev, len(daily)))

for i, (start, end) in enumerate(segments):
    if end - start < 3:
        continue
    date_start = daily.iloc[start]["date"]
    date_end = daily.iloc[min(end, len(daily)-1)]["date"]

    # Članki v tem segmentu
    mask = (sentiment["date"] >= date_start) & (sentiment["date"] < date_end)
    texts = sentiment[mask]["lead"].dropna().tolist()

    if len(texts) < 5:
        continue

    vec = TfidfVectorizer(max_features=10, stop_words=SLO_STOPWORDS, min_df=2)
    try:
        vec.fit_transform(texts)
        top_words = vec.get_feature_names_out()
        avg_sentiment = sentiment[mask]["sentiment_raw"].mean()
        print(f"\nSegment {i+1}: {date_start.date()} → {date_end.date()} "
              f"(sentiment: {avg_sentiment:.2f}, {len(texts)} člankov)")
        print(f"  Top besede: {', '.join(top_words)}")
    except:
        pass

print("\nDone.")
