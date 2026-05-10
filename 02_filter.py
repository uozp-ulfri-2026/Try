import pandas as pd

INPUT = "mmc.parquet"
OUTPUT_SPORT = "sport.parquet"
OUTPUT_SKOKI = "skoki.parquet"

print("Berem parquet...")
df = pd.read_parquet(INPUT)
print(f"Skupaj člankov: {len(df)}")

# Filtriraj samo šport
sport = df[df["topics"] == "sport"].copy()
print(f"Športnih člankov: {len(sport)}")

sport.to_parquet(OUTPUT_SPORT, index=False)
print(f"Shranjeno v {OUTPUT_SPORT}")

# Filtriraj smučarske skoke
KEYWORDS = [
    "smučarski skoki", "smučarskih skokov", "skakalec", "skakalci",
    "skakalnica", "ski jumping", "skiflying",
    "Prevc", "Kraft", "Kobayashi", "Geiger", "Wellinger",
    "Planica", "Willingen", "Oberstdorf", "Innsbruck", "Bischofshofen",
    "Turnej štirih", "posamična tekma v smučarskih",
    "Svetovni pokal v smučarskih"
]

pattern = "|".join(KEYWORDS)
mask = (
    sport["keywords"].str.contains(pattern, case=False, na=False) |
    sport["lead"].str.contains(pattern, case=False, na=False) |
    sport["body"].str.contains(pattern, case=False, na=False)
)
skoki = sport[mask].copy()
print(f"\nČlankov o smučarskih skokih: {len(skoki)}")
print(f"\nVzorec naslovov (lead):")
for lead in skoki["lead"].head(5):
    print(f"  - {lead[:100]}")

skoki.to_parquet(OUTPUT_SKOKI, index=False)
print(f"\nShranjeno v {OUTPUT_SKOKI}")

# Statistika po letih
print(f"\nČlanki po letih:")
print(skoki["date"].dt.year.value_counts().sort_index())

# Preverimo omembe Prevca
prevc_mask = skoki["body"].str.contains("Prevc", case=False, na=False)
print(f"\nČlankov z omembo Prevc: {prevc_mask.sum()}")
