import pandas as pd

df = pd.read_parquet("sport.parquet")
print(f"Skupaj športnih člankov: {len(df)}")

SPORTS = {
    "kosarka": {
        "keywords": ["košarka", "NBA", "Dončić", "Doncic", "Cedevita", "Olimpija"],
        "label": "Košarka"
    },
    "hokej": {
        "keywords": ["hokej", "NHL", "Kopitar", "HDD", "Olimpija", "KHL"],
        "label": "Hokej"
    },
    "alpsko_smucanje": {
        "keywords": ["alpsko smučanje", "slalom", "veleslalom", "smuk", "superveleslalom",
                     "Štuhec", "Stuhec", "Kranjec", "Höfl", "Shiffrin", "Odermatt"],
        "label": "Alpsko smučanje"
    },
    "biatlon": {
        "keywords": ["biatlon", "biathlon", "Fak", "Trčan", "Repinc"],
        "label": "Biatlon"
    },
}

for sport_key, config in SPORTS.items():
    pattern = "|".join(config["keywords"])
    mask = (
        df["keywords"].str.contains(pattern, case=False, na=False) |
        df["lead"].str.contains(pattern, case=False, na=False) |
        df["body"].str.contains(pattern, case=False, na=False)
    )
    result = df[mask].copy()
    out = f"{sport_key}.parquet"
    result.to_parquet(out, index=False)
    print(f"{config['label']}: {len(result)} člankov → {out}")
