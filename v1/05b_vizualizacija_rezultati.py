import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np

# Naloži podatke
sentiment = pd.read_csv("skoki_sentiment.csv", parse_dates=["date"])
sentiment = sentiment.dropna(subset=["date", "sentiment_raw"])
sentiment = sentiment.sort_values("date")

results = pd.read_csv("prevc_results_2026.csv", parse_dates=["date"])

# Dnevno povprečje sentimenta
daily = sentiment.groupby(sentiment["date"].dt.date)["sentiment_raw"].mean().reset_index()
daily.columns = ["date", "sentiment"]
daily["date"] = pd.to_datetime(daily["date"])
daily["rolling"] = daily["sentiment"].rolling(7, center=True).mean()

# Barve po kategoriji
COLOR_MAP = {
    "zmaga":     "#2ecc71",   # zelena
    "stopničke": "#3498db",   # modra
    "top10":     "#f39c12",   # oranžna
    "slabo":     "#e74c3c",   # rdeča
}

fig, ax = plt.subplots(figsize=(16, 6))

# Sentiment
ax.scatter(daily["date"], daily["sentiment"], alpha=0.25, s=12, color="#bbbbbb", zorder=1)
ax.plot(daily["date"], daily["rolling"], color="#2c3e50", linewidth=2, zorder=2, label="7-dnevno drseče povprečje")
ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)

# Rezultati — navpične črte + pike
for _, row in results.iterrows():
    if pd.isna(row["date"]) or pd.isna(row["rank"]):
        continue
    cat = row["category"]
    color = COLOR_MAP.get(cat, "#999999")
    ax.axvline(x=row["date"], color=color, alpha=0.3, linewidth=1.2, zorder=0)
    ax.scatter(row["date"], 0.95, color=color, s=80, zorder=5,
               transform=ax.get_xaxis_transform(), clip_on=False)

# Legenda za rezultate
patches = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items()]
line = plt.Line2D([0], [0], color="#2c3e50", linewidth=2, label="Drseče povprečje")
ax.legend(handles=patches + [line], loc="upper left", fontsize=9)

ax.set_title("Sentiment člankov o smučarskih skokih (RTV MMC 2025–2026)\nz označenimi rezultati Domna Prevca", fontsize=13, pad=12)
ax.set_xlabel("Datum")
ax.set_ylabel("Sentiment (-1 = negativno, +1 = pozitivno)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
ax.set_ylim(-1, 1)

# Omeji prikaz na sezono 2025-2026
ax.set_xlim(pd.Timestamp("2025-10-01"), pd.Timestamp("2026-05-01"))

plt.tight_layout()
plt.savefig("sentiment_z_rezultati.png", dpi=150, bbox_inches="tight")
print("Shranjeno: sentiment_z_rezultati.png")
