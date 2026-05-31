# Medijska Asimetrija

Analiza sentimenta slovenskega športnega poročanja na portalu RTV MMC (2023–2026). Projekt raziskuje, ali mediji reagirajo bolj intenzivno na pozitivne kot negativne športne rezultate.

**Avtor:** Miha Perpar

## Struktura

```
v1/    ← stari pipeline (Streamlit app, osnovna analiza)
v2/    ← nov pipeline (embeddingi, XLM-RoBERTa, D3.js frontend)
oznacevalnik/  ← Flask app za ročno označevanje sentimenta (Docker)
```

## v2 Pipeline

```
mmc.yaml (73k člankov)
  └─ 01_klasifikacija_sportov.py  →  sport_clanki.parquet
       └─ 02_sentiment.py          →  sentiment_xlm_roberta_base_sentiment_multilingual.parquet
            └─ 05_precompute.py    →  data/sports.json + data/sport_{id}.json (×12)
                                       └─ web/   ← D3.js frontend
```

Iterativno izboljšanje klasifikacije:
```
06_oznacevalnik_sportov.py   ← terminal app za ročno označevanje
07_self_training.py          ← LogisticRegression na embeddingih → sport_clanki_trained.parquet
```

## Tehnike

- **Klasifikacija športov** — sentence embeddingi (`paraphrase-multilingual-MiniLM-L12-v2`) + cosine similarity; iterativno izboljšano s self-training logistično regresijo (444 ročnih oznak)
- **Sentiment analiza** — `cardiffnlp/xlm-roberta-base-sentiment-multilingual`, paragraph-level z agregacijo na članek; izbran po primerjavi 3 modelov (Cohen's κ = 0.305)
- **LOWESS glajenje** — fiksno okno K=25 člankov (ne dni) za primerljive krivulje med šport z različno gostoto
- **Zaznava ekstremov** — `scipy.signal.find_peaks` s prominence pragom 8% razpona signala
- **TF-IDF ključne besede** — background IDF na vseh člankih šport, transform na kontekstu ±14 dni; lematizacija z `lemmagen3`

## Frontend

Statični D3.js — brez strežnika, odpri direktno v brskalniku:

```
v2/web/index.html   ← primerjava 12 športov
v2/web/sport.html   ← detail stran posameznega športa
```

Podatki: `v2/data/*.json` (precomputed, ~1.6 MB skupaj).

## Podatki

Surovi podatki (`mmc.yaml`, 322 MB) in parquet datoteke niso v repozitoriju.

## Zagon v2

```bash
python3 -m venv ~/medijska-asimetrija
source ~/medijska-asimetrija/bin/activate
pip install pandas numpy scipy scikit-learn statsmodels sentence-transformers \
            transformers torch lemmagen3 pyyaml pyarrow flask

cd v2
python3 01_klasifikacija_sportov.py --input ../mmc.yaml
python3 02_sentiment.py --model cardiffnlp/xlm-roberta-base-sentiment-multilingual
python3 05_precompute.py
```

Podrobna dokumentacija odločitev: [`v2/PIPELINE.md`](v2/PIPELINE.md)
