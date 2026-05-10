import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Domen Prevc FIS ID: 2278
# https://www.fis-ski.com/DB/general/athlete-biography.html?sectorcode=JP&competitorid=2278

BASE_URL = "https://www.fis-ski.com/DB/general/results.html"
PARAMS = {
    "sectorcode": "JP",
    "competitorid": "2278",
    "type": "result",
}

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

print("Scrapam rezultate Domna Prevca s FIS...")

all_results = []

# Sezone 2023-2026
for season in ["2023", "2024", "2025", "2026"]:
    params = {**PARAMS, "seasonyear": season}
    try:
        r = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        rows = soup.select("table.table-striped tbody tr")
        if not rows:
            # Poskusi alternativni selektor
            rows = soup.select(".g-row.justify-sb")

        print(f"Sezona {season}: najdenih {len(rows)} vrstic v HTML")

        # Shrani raw HTML za debug
        with open(f"fis_raw_{season}.html", "w", encoding="utf-8") as f:
            f.write(r.text)

        time.sleep(1)

    except Exception as e:
        print(f"Napaka za sezono {season}: {e}")

print("\nRaw HTML shranjen kot fis_raw_XXXX.html")
print("Preveri fis_raw_2025.html v brskalniku ali z: python3 03b_parse_fis.py")
