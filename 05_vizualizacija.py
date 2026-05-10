import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("skoki_sentiment.csv", parse_dates=["date"])
df = df.dropna(subset=["date", "sentiment_raw"])
df = df.sort_values("date")

# Dnevno povprečje
daily = df.groupby(df["date"].dt.date)["sentiment_raw"].mean().reset_index()
daily.columns = ["date", "sentiment"]
daily["date"] = pd.to_datetime(daily["date"])

# 7-dnevno drseče povprečje
daily["rolling"] = daily["sentiment"].rolling(7, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 5))

ax.scatter(daily["date"], daily["sentiment"], alpha=0.3, s=15, color="#aaaaaa", label="Dnevno povprečje")
ax.plot(daily["date"], daily["rolling"], color="#e63946", linewidth=2, label="7-dnevno drseče povprečje")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_title("Sentiment člankov o smučarskih skokih (RTV MMC)", fontsize=14, pad=15)
ax.set_xlabel("Datum")
ax.set_ylabel("Sentiment (-1 = negativno, +1 = pozitivno)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.legend()
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig("sentiment_timeline.png", dpi=150, bbox_inches="tight")
print("Shranjeno: sentiment_timeline.png")
print(f"\nObdobje: {daily['date'].min().date()} → {daily['date'].max().date()}")
print(f"Dni s podatki: {len(daily)}")
