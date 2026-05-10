import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Medijska Asimetrija", page_icon="📰", layout="wide")

st.title("Medijska Asimetrija")
st.caption("Sentiment slovenskega športnega poročanja — RTV MMC 2023–2026")

SLO_STOPWORDS = [
    "in", "je", "so", "se", "na", "za", "da", "ki", "v", "z", "s", "ko",
    "pa", "ne", "po", "bi", "ga", "mu", "jo", "jih", "jim", "mi", "me",
    "si", "bo", "bila", "bilo", "bili", "bile", "sem", "smo", "ste",
    "ter", "ali", "kot", "več", "tudi", "le", "že", "še", "od", "do",
    "pri", "med", "nad", "pod", "pred", "ob", "iz", "ta", "to", "te",
    "ti", "tega", "temu", "tem", "tej", "ker", "kar", "kaj", "kdo",
    "kako", "kdaj", "kjer", "če", "toda", "ampak", "a", "o", "ni",
    "niso", "ima", "imajo", "bil", "sta",
]

SPORT_FILES = {
    "Smučarski skoki": "skoki_sentiment.csv",
    "Kolesarstvo": "kolesarstvo_sentiment.csv",
    "Košarka": "kosarka_sentiment.csv",
    "Hokej": "hokej_sentiment.csv",
    "Alpsko smučanje": "alpsko_smucanje_sentiment.csv",
    "Biatlon": "biatlon_sentiment.csv",
}

@st.cache_data
def load_segments():
    with open("segments.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_sentiment(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["date"])
    return df.dropna(subset=["date", "sentiment_raw"]).sort_values("date")

def compute_tfidf(texts, n=8):
    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(max_features=n, stop_words=SLO_STOPWORDS, min_df=1)
    try:
        mat = vec.fit_transform(texts)
        scores = mat.mean(axis=0).A1
        words = vec.get_feature_names_out()
        return sorted(zip(words, scores), key=lambda x: -x[1])
    except:
        return []

segments_data = load_segments()
sports = list(SPORT_FILES.keys())

# --- Sidebar ---
st.sidebar.title("Nastavitve")
selected_sport = st.sidebar.selectbox("Izberi šport", sports)
show_scatter = st.sidebar.checkbox("Prikaži posamezne dneve", value=True)
metric = st.sidebar.radio(
    "Metrika",
    ["Uteženi sentiment", "% pozitivnih − % negativnih"],
    help="Uteženi sentiment: povprečje z utežmi. Ratio: delež pozitivnih minus negativnih člankov."
)

st.sidebar.divider()
st.sidebar.subheader("O modelu")
st.sidebar.markdown(
    "**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`\n\n"
    "**Validacija** (n=30, stratificirano):\n"
    "- Accuracy: **56.7%**\n"
    "- Cohen's κ: **0.350**\n"
    "- Negative recall: 88%\n"
    "- Positive recall: 55%\n\n"
    "Model podcenjuje pozitivni sentiment → "
    "dejanska asimetrija je verjetno večja."
)

sport_key = selected_sport
seg_data = segments_data.get(sport_key, None)
raw_sentiment = load_sentiment(SPORT_FILES[selected_sport])

timeline = pd.DataFrame(seg_data["timeline"])
timeline["date"] = pd.to_datetime(timeline["date"])
segments = seg_data["segments"]
asymmetry = seg_data["asymmetry"]
mean_smoothed = seg_data["mean_smoothed"]

# --- Pregled športa ---
n_total_sport = len(raw_sentiment)
date_min = raw_sentiment["date"].min().strftime("%b %Y")
date_max = raw_sentiment["date"].max().strftime("%b %Y")
pct_pos = (raw_sentiment["sentiment_label"] == "positive").mean() * 100
pct_neg = (raw_sentiment["sentiment_label"] == "negative").mean() * 100
pct_neu = (raw_sentiment["sentiment_label"] == "neutral").mean() * 100

ov1, ov2, ov3, ov4 = st.columns(4)
ov1.metric("Člankov", n_total_sport)
ov2.metric("Obdobje", f"{date_min} – {date_max}")
ov3.metric("Pozitivnih", f"{pct_pos:.0f}%")
ov4.metric("Negativnih", f"{pct_neg:.0f}%")

# --- Tabi ---
tab_sentiment, tab_asymmetry = st.tabs(["📈 Sentiment skozi čas", "⚖️ Asimetrija poročanja"])

# ============================================================
# TAB 1: Sentiment skozi čas
# ============================================================
with tab_sentiment:

    # --- Graf ---
    fig = go.Figure()

    y_col = "smoothed" if metric == "Uteženi sentiment" else "ratio_smoothed"
    y_raw = "sentiment" if metric == "Uteženi sentiment" else "ratio"
    y_label = "Uteženi sentiment" if metric == "Uteženi sentiment" else "% pozitivnih − % negativnih"
    mean_val = float(timeline[y_col].mean())

    if show_scatter:
        fig.add_trace(go.Scatter(
            x=timeline["date"], y=timeline[y_raw],
            mode="markers",
            marker=dict(size=4, color="#cccccc", opacity=0.4),
            name="Dnevna vrednost",
            hovertemplate="%{x|%d. %b %Y}: %{y:.3f}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=timeline["date"], y=timeline[y_col],
        mode="lines",
        line=dict(color="#2c3e50", width=2.5),
        name=f"LOWESS — {y_label}",
        hovertemplate="%{x|%d. %b %Y}: %{y:.3f}<extra></extra>"
    ))

    fig.add_hline(y=mean_val, line_dash="dot", line_color="#aaaaaa",
                  annotation_text="povprečje", annotation_position="bottom right")

    for seg in segments:
        if seg["cp_date"] is None:
            continue
        color = "#27ae60" if seg["above_mean"] else "#e74c3c"
        fig.add_vline(x=seg["cp_date"], line_dash="dash",
                      line_color=color, line_width=1.2, opacity=0.5)
        cp_dt = pd.to_datetime(seg["cp_date"])
        closest_idx = (timeline["date"] - cp_dt).abs().idxmin()
        cp_y = float(timeline.loc[closest_idx, y_col])
        fig.add_trace(go.Scatter(
            x=[seg["cp_date"]], y=[cp_y],
            mode="markers",
            marker=dict(size=10, color=color, line=dict(width=1.5, color="white")),
            showlegend=False,
            hovertemplate=f"<b>{seg['cp_date']}</b><br>sentiment: {seg['avg_sentiment']:.3f}<br>{seg['n_articles']} člankov<extra></extra>"
        ))

    fig.update_layout(
        height=400,
        dragmode="select",
        selectdirection="h",
        xaxis_title="Datum",
        yaxis_title=y_label,
        yaxis_range=[-0.3, 0.6],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0)
    )

    selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    # --- Izbira obdobja: drag-select ALI segment dropdown ---
    date_from = None
    date_to = None

    # Drag-select iz grafa
    if selected and selected.get("selection") and selected["selection"].get("box"):
        box = selected["selection"]["box"]
        if box and len(box) > 0:
            x_range = box[0].get("x", [])
            if len(x_range) == 2:
                date_from = pd.to_datetime(x_range[0])
                date_to = pd.to_datetime(x_range[1])

    # Segment selector kot fallback
    if not date_from:
        segment_options = ["— Izberi segment —"]
        segment_map = {}
        for seg in segments:
            label_icon = "🟢" if seg["above_mean"] else "🔴"
            label = f"{label_icon} {seg['date_start']} → {seg['date_end']}  ({seg['n_articles']} čl., sentiment {seg['avg_sentiment']:+.3f})"
            segment_options.append(label)
            segment_map[label] = seg

        chosen = st.selectbox(
            "Ali izberi segment iz seznama:",
            segment_options, label_visibility="collapsed"
        )
        if chosen != "— Izberi segment —":
            seg = segment_map[chosen]
            date_from = pd.to_datetime(seg["date_start"])
            date_to = pd.to_datetime(seg["date_end"])

    # --- Analiza izbranega obdobja ---
    if date_from and date_to:
        mask = (raw_sentiment["date"] >= date_from) & (raw_sentiment["date"] <= date_to)
        period_df = raw_sentiment[mask].copy()

        st.markdown(f"### Obdobje: {date_from.strftime('%d. %b %Y')} → {date_to.strftime('%d. %b %Y')}")

        if len(period_df) == 0:
            st.warning("V izbranem obdobju ni člankov.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Člankov", len(period_df))
            col2.metric("Povp. sentiment", f"{period_df['sentiment_raw'].mean():.3f}")
            col3.metric("% pozitivnih", f"{(period_df['sentiment_label'] == 'positive').mean()*100:.1f}%")

            # Stacked bar: izbrano obdobje vs celotni šport
            n_pos = (period_df["sentiment_label"] == "positive").sum()
            n_neu = (period_df["sentiment_label"] == "neutral").sum()
            n_neg = (period_df["sentiment_label"] == "negative").sum()
            n_total = len(period_df)

            n_pos_all = (raw_sentiment["sentiment_label"] == "positive").sum()
            n_neu_all = (raw_sentiment["sentiment_label"] == "neutral").sum()
            n_neg_all = (raw_sentiment["sentiment_label"] == "negative").sum()
            n_total_all = len(raw_sentiment)

            fig_bar = go.Figure()
            for label_name, color, n_sel, n_all in [
                ("Pozitivni", "#27ae60", n_pos, n_pos_all),
                ("Nevtralni", "#95a5a6", n_neu, n_neu_all),
                ("Negativni", "#e74c3c", n_neg, n_neg_all),
            ]:
                fig_bar.add_trace(go.Bar(
                    name=label_name,
                    x=[n_sel/n_total*100, n_all/n_total_all*100],
                    y=["Izbrano obdobje", "Celotni šport"],
                    orientation="h", marker_color=color,
                    text=[f"{n_sel/n_total*100:.0f}%", f"{n_all/n_total_all*100:.0f}%"],
                    textposition="inside",
                ))
            fig_bar.update_layout(
                barmode="stack", height=140,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 100], showticklabels=False),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            col_art, col_tfidf = st.columns([1, 1])

            with col_art:
                title_col = "title" if "title" in period_df.columns else "lead"

                st.markdown("**🟢 Najbolj pozitivni članki:**")
                top_pos = period_df.nlargest(3, "sentiment_raw")
                for i, (_, row) in enumerate(top_pos.iterrows(), 1):
                    text = str(row[title_col])[:110] if pd.notna(row[title_col]) else ""
                    score = f"`{row['sentiment_raw']:+.3f}`"
                    url = row.get("url", "")
                    if pd.notna(url) and url:
                        st.markdown(f"{i}. [{text}]({url}) {score}")
                    else:
                        st.markdown(f"{i}. {text} {score}")

                st.markdown("**🔴 Najbolj negativni članki:**")
                top_neg = period_df.nsmallest(3, "sentiment_raw")
                for i, (_, row) in enumerate(top_neg.iterrows(), 1):
                    text = str(row[title_col])[:110] if pd.notna(row[title_col]) else ""
                    score = f"`{row['sentiment_raw']:+.3f}`"
                    url = row.get("url", "")
                    if pd.notna(url) and url:
                        st.markdown(f"{i}. [{text}]({url}) {score}")
                    else:
                        st.markdown(f"{i}. {text} {score}")

            with col_tfidf:
                st.markdown("**Ključne besede (TF-IDF):**")
                texts = period_df["lead"].dropna().tolist()
                tfidf = compute_tfidf(texts)
                if tfidf:
                    words, scores = zip(*tfidf)
                    fig_t = go.Figure(go.Bar(
                        x=list(scores), y=list(words),
                        orientation="h",
                        marker_color="#3b82f6"
                    ))
                    fig_t.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("⬆ Povleci z miško po grafu ali izberi segment iz seznama za podrobno analizo.")

# ============================================================
# TAB 2: Asimetrija poročanja
# ============================================================
with tab_asymmetry:

    st.markdown(
        "Asimetrija meri razliko med **visokimi** (nad povprečjem) in **nizkimi** "
        "(pod povprečjem) segmenti LOWESS krivulje. Večja razlika pomeni, da mediji "
        "reagirajo bolj intenzivno na uspehe kot na neuspehe."
    )

    # Dumbbell chart — primerjava med športi
    st.markdown("### Primerjava med športi")
    st.caption("🟢 Visoki segmenti &nbsp;&nbsp; 🔴 Nizki segmenti &nbsp;&nbsp; Daljša črta = večja asimetrija")

    asym_rows = []
    for sport_name, sdata in segments_data.items():
        asym_rows.append({
            "šport": sport_name,
            "visoki": sdata["asymmetry"]["above_mean"],
            "nizki": sdata["asymmetry"]["below_mean"],
            "razlika": sdata["asymmetry"]["difference"],
        })
    asym_df = pd.DataFrame(asym_rows).sort_values("razlika", ascending=True)

    fig_db = go.Figure()

    # Connector lines
    for _, row in asym_df.iterrows():
        is_selected = row["šport"] == selected_sport
        fig_db.add_trace(go.Scatter(
            x=[row["nizki"], row["visoki"]],
            y=[row["šport"], row["šport"]],
            mode="lines",
            line=dict(color="#f39c12" if is_selected else "#888888", width=3 if is_selected else 2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Nizki dots (red)
    fig_db.add_trace(go.Scatter(
        x=asym_df["nizki"], y=asym_df["šport"],
        mode="markers",
        marker=dict(size=12, color="#e74c3c", line=dict(width=1.5, color="white")),
        name="Nizki segmenti",
        hovertemplate="%{y}: %{x:.3f}<extra>Nizki</extra>",
    ))

    # Visoki dots (green)
    fig_db.add_trace(go.Scatter(
        x=asym_df["visoki"], y=asym_df["šport"],
        mode="markers",
        marker=dict(size=12, color="#27ae60", line=dict(width=1.5, color="white")),
        name="Visoki segmenti",
        hovertemplate="%{y}: %{x:.3f}<extra>Visoki</extra>",
    ))

    # Razlika annotations
    for _, row in asym_df.iterrows():
        fig_db.add_annotation(
            x=row["visoki"] + 0.008, y=row["šport"],
            text=f"Δ {row['razlika']:.3f}",
            showarrow=False,
            font=dict(size=12, color="#f39c12"),
            xanchor="left",
        )

    fig_db.update_layout(
        height=50 + len(asym_df) * 55,
        xaxis_title="Povp. sentiment",
        xaxis=dict(range=[0, max(asym_df["visoki"].max(), 0.2) + 0.04]),
        margin=dict(l=0, r=80, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_db, use_container_width=True)

    # Detail za izbrani šport
    st.divider()
    st.markdown(f"### {selected_sport}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Visoki segmenti", f"{asymmetry['above_mean']:.3f}",
                f"{asymmetry['n_above']} segmentov")
    col2.metric("Nizki segmenti", f"{asymmetry['below_mean']:.3f}",
                f"{asymmetry['n_below']} segmentov")
    diff_val = asymmetry["difference"]
    col3.metric("Razlika (Δ)", f"{diff_val:.3f}")

    if diff_val > 0.05:
        st.success("**Močna asimetrija** — mediji izrazito bolj pozitivno poročajo o uspehih kot neuspehih.")
    elif diff_val > 0.02:
        st.warning("**Zmerna asimetrija** — opazna razlika med visokimi in nizkimi segmenti.")
    else:
        st.info("**Šibka asimetrija** — poročanje je relativno uravnoteženo.")
