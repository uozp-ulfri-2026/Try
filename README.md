# Medijska Asimetrija

Analiza sentimenta slovenskega športnega poročanja na portalu RTV MMC (2023–2026). Projekt raziskuje, ali mediji reagirajo bolj intenzivno na pozitivne kot negativne športne rezultate.

**Ekipa:** Try — Miha Perpar

## Ključne ugotovitve

Biatlon in smučarski skoki kažeta največjo asimetrijo v poročanju — pozitivni rezultati dobijo nesorazmerno več pozitivnega pokritja. Model podcenjuje pozitivni sentiment (accuracy 56.7%, κ = 0.350), kar pomeni da je dejanska asimetrija verjetno še večja.

| Šport | Visoki seg. | Nizki seg. | Razlika (Δ) |
|-------|------------|-----------|-------------|
| Biatlon | 0.136 | 0.044 | 0.092 |
| Smučarski skoki | 0.176 | 0.109 | 0.067 |
| Alpsko smučanje | 0.095 | 0.069 | 0.026 |
| Košarka | 0.071 | 0.043 | 0.026 |
| Kolesarstvo | 0.068 | 0.042 | 0.026 |
| Hokej | 0.058 | 0.039 | 0.020 |

## Tehnike

- **Sentiment analiza** — XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`), večjezični model treniran na družbenih medijih
- **LOWESS glajenje** — nelokalna regresija (frac=0.08) za trend sentimenta skozi čas
- **Detekcija obratov** — lokalni ekstremi na LOWESS krivulji (`scipy.signal.argrelextrema`)
- **TF-IDF** — ključne besede po segmentih za strojne razlage

## Podatki

Surovi podatki (`mmc.yaml`, 123MB, 73.461 člankov) niso v repozitoriju. Sentiment CSV datoteke za 6 športov in `segments.json` so vključeni.

## Pipeline

```
01_yaml_to_parquet.py    YAML → Parquet
02_filter.py             Filter športnih člankov
08_filter_sports.py      Filter posameznih športov
04_sentiment.py          Sentiment analiza z XLM-RoBERTa
07_precompute.py         LOWESS + changepoints + TF-IDF → segments.json
06_changepoints.py       Statični PNG grafi
```

## Streamlit app

```bash
source ~/medijska-asimetrija/bin/activate
streamlit run app.py
```

Interakcije: izbira športa, drag-select obdobja, segment selector, primerjava z celotnim športom, dumbbell chart asimetrije, klikljivi članki.

## Validacija

```bash
python3 10_oznaci.py validacija_slepa.csv    # interaktivno označevanje
python3 09_validacija.py validacija_slepa.csv validacija_kljuc.csv   # metrike
```

## Instalacija

```bash
python3 -m venv ~/medijska-asimetrija
source ~/medijska-asimetrija/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas pyarrow pyyaml scikit-learn matplotlib \
            ruptures statsmodels streamlit plotly scipy
```
