# Pipeline v2 — Dokumentacija

Analiza sentimenta slovenskega športnega poročanja (RTV MMC 2023–2026).  
Vhod: `mmc.yaml` (73.461 člankov, ~320 MB) → Izhod: `data/*.json` za frontend.

---

## Korak 1 — Klasifikacija športov (`01_klasifikacija_sportov.py`)

### Problem s staro verzijo
Stara verzija (`02_filter.py`) je uporabljala hardkodirane keywordie (seznam imen atletov, prizorišč...). Ni se posplošilo na nove športe, vzdrževanje je bilo ročno.

### Pristop: multilingual sentence embeddings
Model: `paraphrase-multilingual-MiniLM-L12-v2`

Za vsak šport definiramo "sidro" — kratko slovensko besedilo s ključnimi pojmi. Vsak članek embeddamo (title + lead + keywords), potem izračunamo cosine similarity med člankom in vsakim sidrom. Članek dobi šport z najvišjo similarnostjo.

**Zakaj ta model in ne zero-shot klasifikacija:**
- 10× hitrejši (MiniLM je majhen, ~90MB)
- GPU ni potreben za inference
- Za klasifikacijo po temi deluje enako dobro

**Hitrost na RTX 2060:** ~5–15 sekund za embedding 17.5k člankov (po nalaganju YAML).

### Problemi pri prvem zagonu

**Curling (339 člankov)** — zajel je šah, judo, kolesarstvo. Krivec: beseda `list` v sidru (`curling list` = curling steza), ki je v slovenščini izjemno pogosta beseda.

**Namizni tenis (543 člankov)** — zajel je navaden tenis. Problem: oba delita isto besedo `tenis`, model ni ločeval.

**Popravek:** Odstranili smo generične fraze (`svetovno prvenstvo`, `olimpijske igre`) ki so skupne vsem športom, in iz curling sidra odstranili `list`. Po popravku: Curling 339 → 8, Namizni tenis 543 → 150, Tenis dobil 197 lastnih člankov.

### Prag confidence
Članki pod pragom 0.25 gredo v kategorijo "Drugo" (5308 člankov = ~30%).

### Izbrani športi (200+ člankov)
| Šport | Člankov |
|---|---|
| Nogomet | 1885 |
| Rokomet | 1729 |
| Alpsko smučanje | 1699 |
| Kolesarstvo | 1380 |
| Košarka | 999 |
| Hokej na ledu | 631 |
| Atletika | 397 |
| Smučarski skoki | 360 |
| Odbojka | 334 |
| Športno plezanje | 274 |
| Biatlon | 199 |
| Tenis | 197 |

---

## Korak 2 — Sentiment analiza (`02_sentiment.py`)

### Pristop: paragraph-level sentiment + agregacija

Namesto sentimenta samo na leadu, razdelimo vsak članek na odstavke in naredimo sentiment na vsakem posebej. Agregacija na nivo članka: `sentiment_raw` = aritmetična sredina odstavkov, `sentiment_label` = majority vote.

**Zakaj paragraph-level:** lead je pogosto kratek in ne zajame celotnega tona članka. Jedro in zaključek nosita dodaten sentiment signal.

### Filtriranje odstavkov
Preskočimo odstavke ki:
- Imajo manj kot 10 besed
- Vsebujejo boilerplate (`preberite tudi`, `foto:`, `©`, ...)
- Imajo več kot 25% besed v CAPS — to so odstavki z rezultati (npr. `KRKA – CIBONA 92:97 (14:28, ...)`)

### MAX_CHARS: 512 → 2000
Prvotno smo rezali na 512 znakov (~70–80 besed). Model podpira 512 **tokenov** (~300–400 besed). Popravek na 2000 znakov (~350 besed) — brez izgube vsebine, `positive` narastel za ~6%.

### Trije testirani modeli

| Model | positive | neutral | negative | κ (validacija) |
|---|---|---|---|---|
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | 80% | **0%** | 20% | 0.130 |
| `cardiffnlp/xlm-roberta-base-sentiment-multilingual` | 43% | 52% | 6% | **0.305** |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` (stari) | 26% | 69% | 5% | 0.143 |

**lxyuan:** neutral = 2 od 10k člankov. Model praktično ne pozna nevtralnega razreda. Neuporabno.

**twitter-xlm:** slabši kot pričakovano (κ=0.143 vs κ=0.350 v stari validaciji z n=30). Razlog: stara validacija je bila le 30 vzorcev, nova je 121. Negative recall samo 17%.

**Izbran model: `cardiffnlp/xlm-roberta-base-sentiment-multilingual`**
- Edini z realistično porazdelitvijo razredov
- Konsistenten z drugimi modeli pri rangiranju športov (Odbojka dosledno najnižja, Smučarski skoki visoki)
- Opomba: negative recall je 35% — model podcenjuje negativni sentiment, dejanska asimetrija je verjetno večja

**Disk space problem:** `xlm-roberta-base-sentiment-multilingual` je 1.1 GB, potrebuje prosto mesto za download.

### Validacija (n=121 ročnih oznak)
Oznake zbrane z oznacevalnikom (Flask web app, dostopen na telefonu).
- 121 od 193 oznak se je ujemalo z ID-ji v sentiment parquetu (72 je bilo v kategoriji "Drugo")
- Distribucija oznak: neutral 87, positive 71, negative 35

---

## Korak 3 — Utežena regresija (`04_utezena_regresija.py`) — eksperiment

### Ideja
Namesto enake teže za vse odstavke, naučimo optimalne uteži za:
- `lead` — uvodni povzetek
- `first` — prvi odstavek
- `middle` — vmesni odstavki (povprečje)
- `last` — zadnji odstavek

Uteži naučimo z Ridge regresijo na 121 ročnih oznakah, 5-fold cross-validation.

### Rezultati
```
Baseline κ (enake uteži):  0.308
5-fold CV κ:               0.317 ± 0.221
```

**Niso se naučile stabilne uteži.** Varianca ±0.221 je ogromna — foldi grejo od 0.067 do 0.725. Razlog: 121 vzorcev / 5 foldov = 24 vzorcev na testni fold, premalo za stabilno oceno.

**Zanimiv vzorec uteži:**
- `last`: 0.035 — zadnji odstavek je skoraj brez vrednosti (kontekst/background)
- `middle`: 0.450 — verjetno artefakt: middle je povprečje več odstavkov → manj šuma → videti bolj prediktivno

**Sklep:** Izboljšava je v mejah šuma. Z 300+ oznakami bi bilo smiselno ponoviti. Za zdaj nadaljujemo s preprostim povprečjem.

---

## Korak 4 — Precompute JSON (`05_precompute.py`)

### Datumski filter
Članki pred `2023-04-28` so izločeni. `date_range[0]` v JSON-u je `"2023-04-28"`, `date_range[1]` je `"2026-03-31"`. Timeline (120 točk) in LOWESS sta izračunana samo na tem intervalu.

### Keywords — TF-IDF z ozadjem sporta
Keywords za vsak ekstrem se računajo z dvostopenjskim TF-IDF:
1. **Fit na vseh člankih šport** → IDF penalizira besede ki so vedno prisotne (ime šport, stalne besede, stalni igralci)
2. **Transform samo kontekstnih člankov** (±14 dni okoli ekstrema) → ostanejo samo besede ki so za to obdobje edinstvene

**Zakaj:** navaden TF-IDF samo na kontekstu bi dal "Planica", "Prevc" pri vsakem ekstremu za smučarske skoke — ker so te besede pogoste v VSEH člankih tega šport, ne samo v tem ekstremu. Z IDF iz ozadja te besede dobijo nizek score, izstopijo pa tiste ki so specifične za to dogajanje.

**Lematizacija:** `lemmagen3` (Slovenian) pretvori sklanjane oblike v osnovno obliko (`iger` → `igra`, `tekmovalcev` → `tekmovalec`).

**Normalizacija uteži:** [0.25, 1.0] — zagotavlja vsaj 4× vizualno razliko med prvo in zadnjo besedo v word cloudu.

**Filtri:** števila, besede ≤ 2 znaka, razširjen seznam slovenskih stopwords (predlogi, vezniki, meseci, splošni glagoli).

Producira JSON datoteke za frontend.

### Metode
- **LOWESS glajenje** — fiksno okno `K=25 člankov` (ne frakcija), 200 izhodnih točk z datumi (`smoothed`)
- **Robno odrezanje** — prvih in zadnjih `frac/2` točk krivulje je odrezanih (LOWESS je na robovih nezanesljiv zaradi enostranskega okna)
- **Timeline** — 120 enakomernih točk na skupni časovni osi za vse športe (pogoj za primerjavo)
- **Ekstremi** — `scipy.signal.find_peaks` s prominence pragom 8% razpona signala; vrh velja samo če je nad povprečjem, dolina samo če je pod (preprečuje zaznavo dipa znotraj peaka)
- **Filter nevtralnih** — članki z `sentiment_label == "neutral"` so izločeni pred računanjem krivulje
- **Keywords** — TF-IDF z IDF iz celotnega sporta (ozadje), TF iz kontekstnih člankov (±14 dni); lematizacija z `lemmagen3`; normalizacija uteži na [0.25, 1.0]

### Zakaj K=25 (fiksno število člankov) namesto fiksnih dni
Z fiksnim oknom v dnevih bi vsaka točka krivulje za football temeljila na 50+ člankih, za biatlon pa na 5 — različna statistična zanesljivost. K=25 zagotavlja da je vsaka točka na krivulji povprečje enakega števila člankov, ne glede na šport. Posledica: football (gost) ima časovno ločljivost ~2 tedni, biatlon (redek) ~2 meseca — krivulji sta enako zanesljivi, samo z različno časovno resolucijo. Za research question (asimetrija sentimenta, ne timing) je to sprejemljivo.

Opomba: proporcionalen K (K = c·n) bi bil ekvivalenten fiksni frakciji in bi vrnil izvorni problem ravnih krivulj za gost dataset.

### Output
```
data/sports.json              # landing page
data/sport_{id}.json          # ena datoteka na šport (12 datotek)
```

Sport IDji: `nogomet`, `rokomet`, `alpsko-smucanje`, `kolesarstvo`, `kosarka`, `hokej-na-ledu`, `atletika`, `smucarski-skoki`, `odbojka`, `sportno-plezanje`, `biatlon`, `tenis`

---

## Celotni pipeline

```
mmc.yaml
  └─ 01_klasifikacija_sportov.py  →  sport_clanki.parquet
       └─ 02_sentiment.py          →  sentiment_xlm_roberta_base_sentiment_multilingual.parquet
                                       sentiment_xlm_roberta_base_sentiment_multilingual_paragraphs.parquet
            └─ 05_precompute.py    →  data/sports.json
                                       data/sport_{id}.json  (×12)
```

Validacija (neodvisno):
```
oznacevalnik/  →  oznake.csv
  └─ 03_validacija.py   (primerja modele)
  └─ 04_utezena_regresija.py  (eksperiment z utežmi)
```

---

---

## Frontend (`web/`)

Statični HTML/JS — brez build procesa, dela v brskalniku direktno iz datotečnega sistema.

**`web/index.html`** — landing page:
- Seznam 12 športov z sparkline grafom in boxplotom
- Primerjalni panel: izberi do 5 športov, prikaz krivulj + boxplotov skupaj
- Y-os: auto (relativno) ali ±1 (absolutno) za primerjavo
- Bere iz `../data/sports.json`

**`web/sport.html`** — detail stran (dostop prek `?id=smucarski-skoki`):
- Glavni graf: LOWESS krivulja + article dots (zeleni/rdeči/sivi po sentimentu)
- Levo: seznam ekstremov z datumom in amplitudo
- Ko klikneš ekstrem: kontekstno okno ±14 dni, word cloud, top 7 člankov po sentimentu
- Tooltip ob hoverju na dot, klik odpre MMC URL
- Bere iz `../data/sport_{id}.json` + `../data/sports.json`

**Podatki:** `v2/data/*.json` — generira `05_precompute.py`. Frontend bere relativno pot `../data/`.

---

## Zagon od začetka

```bash
source ~/medijska-asimetrija/bin/activate
cd v2

# Osnova (počasi — YAML + embeddingi)
python3 01_klasifikacija_sportov.py --input ../mmc.yaml

# Sentiment (počasi — GPU inference)
python3 02_sentiment.py --model cardiffnlp/xlm-roberta-base-sentiment-multilingual

# Precompute JSON za frontend
python3 05_precompute.py
```

**Z self-trainingom (boljša klasifikacija):**
```bash
# 1. Označi članke
python3 06_oznacevalnik_sportov.py

# 2. Nauči klasifikator (NE prepiše originala)
python3 07_self_training.py
# → sport_clanki_trained.parquet

# 3. Sentiment na novi klasifikaciji
python3 02_sentiment.py \
  --input sport_clanki_trained.parquet \
  --model cardiffnlp/xlm-roberta-base-sentiment-multilingual

# 4. Precompute
python3 05_precompute.py

# 5. Nadaljuj oznacevanje z izboljšanimi predlogi
python3 06_oznacevalnik_sportov.py --input sport_clanki_trained.parquet
```

---

---

## Korak 5 — Ročno označevanje športov (`06_oznacevalnik_sportov.py`)

### Problem s klasifikacijo
Embedding-based klasifikacija (korak 1) ima ~10–15% crossover med podobnimi športi (Rokomet/Košarka/Odbojka, Alpsko smučanje/Smučarski skoki). Rešitev: self-training — ročno popraviti napačno klasificirane članke in naučiti boljši klasifikator.

### Terminal app za označevanje
```
python3 06_oznacevalnik_sportov.py                     # negotovi (conf ≤ 0.55)
python3 06_oznacevalnik_sportov.py --sport Rokomet     # samo en šport
python3 06_oznacevalnik_sportov.py --all               # vsi članki
```

Prikazuje članke urejene po `conf_gap` (razlika med top-1 in top-2 similarnostjo) — najnegotovejši najprej. Brez `conf_gap` stolpca (star parquet) sortira po `confidence`.

Ukazi: `[enter]` potrdi, `[1–13]` popravi šport, `[s]` skip, `[u]` undo zadnji vnos, `[q]` shrani in izhod.

Oznake se shranjujejo v `sport_labels.csv` (append ob vsakem vnosu). Ob ponovnem zagonu preskoči že označene.

**`conf_gap` stolpec:** dodan v `01_klasifikacija_sportov.py` — poleg `confidence` (top-1) shrani tudi `top2_sport`, `top2_conf`, `top3_sport`, `top3_conf`, `conf_gap` (top1−top2). Zahteva ponovni zagon klasifikatorja.

---

## Korak 6 — Self-training klasifikator (`07_self_training.py`)

### Pristop
1. Naloži ročne oznake iz `sport_labels.csv`
2. Dodaj pseudo-labele iz originalne klasifikacije (conf ≥ 0.65) — pokrijejo športe ki niso zastopani v ročnih oznakah
3. Ročne oznake prepišejo pseudo-labele za iste članke
4. Nauči `LogisticRegression` z 5-fold CV na sentence embeddingih
5. Embedaj VSE članke → prediciraj šport → shrani v `sport_clanki_trained.parquet`

### Zakaj pseudo-labeli
Ročne oznake prihajajo samo iz negotovih člankov (nizka confidence) → niso reprezentativni vzorec vseh športov. Brez pseudo-labelov classifier ne vidi npr. Rokometa ali Odbojke in jih ne zna napovedati → vse gre v Nogomet.

### Kritična napaka pri prvem zagonu
Prvič je script prepisal `sport_clanki.parquet` z napačno klasifikacijo (samo Nogomet/Košarka). Vzrok: manjkali pseudo-labeli. Popravek:
- Script zdaj piše v `sport_clanki_trained.parquet` (originalni se nikoli ne dotakne)
- `--parquet` = vhod (read-only), `--output` = izhod (default: `sport_clanki_trained.parquet`)

### Iterativni workflow
```
06_oznacevalnik_sportov.py          # označi
07_self_training.py                 # re-train
06_oznacevalnik_sportov.py \
  --input sport_clanki_trained.parquet  # nadaljuj z boljšimi predlogi
```

### Output
```
sport_clanki_trained.parquet   # nova klasifikacija (ne prepiše originala)
```

Za downstream:
```bash
python3 02_sentiment.py --input sport_clanki_trained.parquet
python3 05_precompute.py
```

---

## Odprte točke

- **Fine-tuning:** z 300+ ročnih oznak bi fine-tuning na `xlm-roberta-base-sentiment-multilingual` verjetno dvignil κ nad 0.4
- **Utežena regresija:** ponoviti ko bo na voljo več oznak
- **"Drugo" kategorija:** 30% športnih člankov ni klasificiranih — nekateri pokrivajo več športov hkrati
- **conf_gap v parquetu:** zahteva ponovni zagon `01_klasifikacija_sportov.py`; brez tega app sortira po `confidence`
