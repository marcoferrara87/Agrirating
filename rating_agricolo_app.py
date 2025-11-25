import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. Metadati e parametri fittizi
# ------------------------------------------------

CROPS = ["Grano duro", "Mais", "Vite", "Olivo", "Ortofrutta", "Bovini latte"]

# Punti interni per regione (niente mare)
REGION_POINTS = {
    "Lazio": [
        (42.40, 12.85),  # area Rieti
        (41.90, 12.60),  # area Roma interna
    ],
    "Campania": [
        (41.13, 14.78),  # Benevento
        (40.93, 14.80),  # Avellino
    ],
    "Puglia": [
        (41.47, 15.55),  # Foggia
        (40.83, 16.55),  # Altamura
    ],
    "Emilia-Romagna": [
        (44.70, 11.00),  # tra Modena e Bologna
        (44.90, 11.40),
    ],
    "Lombardia": [
        (45.60, 9.80),   # Bergamo/Brescia interno
        (45.50, 9.30),   # nord Milano
    ],
    "Toscana": [
        (43.77, 11.25),  # Firenze
        (43.32, 11.33),  # Siena
    ],
}

COLOR_MAP = {
    "A – Basso rischio": [0, 150, 0],
    "B – Rischio moderato": [150, 150, 0],
    "C – Rischio elevato": [200, 120, 0],
    "D – Rischio molto elevato": [200, 0, 0],
}


# ------------------------------------------------
# 2. Generazione dati fittizi
# ------------------------------------------------

def generate_dummy_data(n_farms: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    regions = list(REGION_POINTS.keys())

    farm_ids = [f"FARM-{i:04d}" for i in range(1, n_farms + 1)]
    denominazioni = [f"Azienda Agricola {i}" for i in range(1, n_farms + 1)]
    regioni = np.random.choice(regions, n_farms)
    colture = np.random.choice(CROPS, n_farms)

    superfici = np.round(np.random.uniform(5, 250, n_farms), 1)
    rese = np.round(np.random.uniform(2.0, 12.0, n_farms), 1)
    eco = np.random.randint(0, 6, n_farms)
    comp = np.random.randint(0, 11, n_farms)
    inademp = np.random.randint(0, 4, n_farms)

    price_map = {
        "Grano duro": 270,
        "Mais": 220,
        "Vite": 550,
        "Olivo": 450,
        "Ortofrutta": 300,
        "Bovini latte": 400,
    }
    cost_ha_map = {
        "Grano duro": 700,
        "Mais": 850,
        "Vite": 2500,
        "Olivo": 1800,
        "Ortofrutta": 3000,
        "Bovini latte": 2200,
    }

    df = pd.DataFrame({
        "farm_id": farm_ids,
        "denominazione": denominazioni,
        "regione": regioni,
        "superficie_ha": superfici,
        "coltura_principale": colture,
        "rese_t_ha": rese,
        "eco_schemi_score": eco,
        "compliance_score": comp,
        "anni_inadempienze_ultimi5": inademp,
    })

    df["prezzo_t"] = df["coltura_principale"].map(price_map)
    df["costo_ha"] = df["coltura_principale"].map(cost_ha_map)

    df["ricavi_mercato_eur"] = (
        df["superficie_ha"] * df["rese_t_ha"] * df["prezzo_t"]
    ).round(0)

    pac_ha_base = 250
    pac_ha_eco_bonus = 60
    df["pagamenti_pubblici_eur"] = (
        df["superficie_ha"] * (pac_ha_base + df["eco_schemi_score"] * pac_ha_eco_bonus)
    ).round(0)

    df["costi_operativi_eur"] = (
        df["superficie_ha"] * df["costo_ha"] *
        np.random.uniform(0.9, 1.2, len(df))
    ).round(0)

    df["debito_finanziario_eur"] = (
        df["ricavi_mercato_eur"] * np.random.uniform(0.2, 1.8, len(df))
    ).round(0)

    # Coordinate interne per regione (più sparse, niente mare)
    lats = []
    lons = []
    for reg in df["regione"]:
        pts = REGION_POINTS[reg]
        base_lat, base_lon = pts[np.random.randint(len(pts))]
        lat_j = np.random.normal(0, 0.08)
        lon_j = np.random.normal(0, 0.08)
        lats.append(base_lat + lat_j)
        lons.append(base_lon + lon_j)
    df["lat"] = lats
    df["lon"] = lons

    return df


# ------------------------------------------------
# 3. Motore di rating
# ------------------------------------------------

def compute_financial_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ricavi_totali_eur"] = df["ricavi_mercato_eur"] + df["pagamenti_pubblici_eur"]
    df["ebitda_eur"] = df["ricavi_totali_eur"] - df["costi_operativi_eur"]

    ebitda_floor = 1_000
    df["ebitda_adj_eur"] = df["ebitda_eur"].clip(lower=ebitda_floor)

    df["ebitda_margin"] = (df["ebitda_eur"] / df["ricavi_totali_eur"]).clip(-0.5, 0.6)
    df["rapporto_sussidi"] = (
        df["pagamenti_pubblici_eur"] / df["ricavi_totali_eur"]
    ).clip(0, 0.8)
    df["debito_su_ebitda"] = (
        df["debito_finanziario_eur"] / df["ebitda_adj_eur"]
    ).clip(0, 12)

    df["score_tecnico_ambientale"] = (
        0.6 * (df["eco_schemi_score"] / 5) +
        0.4 * (df["compliance_score"] / 10)
    )
    df["penale_inadempienze"] = df["anni_inadempienze_ultimi5"] * 0.1
    return df


def compute_rating(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_financial_drivers(df)
    score = np.zeros(len(df))

    score += np.interp(df["ebitda_margin"], [-0.2, 0.0, 0.2, 0.4],
                       [0, 20, 30, 35]).clip(0, 35)
    score += np.interp(df["debito_su_ebitda"], [0, 2, 4, 8, 12],
                       [30, 25, 20, 10, 0]).clip(0, 30)
    score += np.interp(df["rapporto_sussidi"], [0.0, 0.2, 0.4, 0.7, 0.8],
                       [15, 15, 10, 5, 0]).clip(0, 15)
    score += df["score_tecnico_ambientale"] * 15
    score -= df["penale_inadempienze"] * 25

    df["score_rischio"] = (
        pd.to_numeric(score, errors="coerce")
        .clip(0, 100)
        .round(1)
    )

    def class_from_score(s):
        if pd.isna(s):
            return "ND"
        if s >= 80:
            return "A – Basso rischio"
        elif s >= 65:
            return "B – Rischio moderato"
        elif s >= 50:
            return "C – Rischio elevato"
        else:
            return "D – Rischio molto elevato"

    df["classe_rating"] = df["score_rischio"].apply(class_from_score)

    spiegazioni = []
    for _, r in df.iterrows():
        parts = []
        if r["ebitda_margin"] > 0.25:
            parts.append("buona redditività operativa")
        elif r["ebitda_margin"] < 0.05:
            parts.append("redditività operativa debole")

        if r["debito_su_ebitda"] <= 3:
            parts.append("indebitamento contenuto")
        elif r["debito_su_ebitda"] > 6:
            parts.append("indebitamento elevato")

        if r["score_tecnico_ambientale"] >= 0.7:
            parts.append("buon profilo tecnico–ambientale")
        elif r["score_tecnico_ambientale"] < 0.4:
            parts.append("profilo tecnico–ambientale critico")

        if r["anni_inadempienze_ultimi5"] > 0:
            parts.append("presenza di inadempienze pregresse")

        if not parts:
            spiegazioni.append("Profilo equilibrato, senza criticità evidenti.")
        else:
            spiegazioni.append("; ".join(parts) + ".")

    df["spiegazione_rating"] = spiegazioni
    return df


def build_company_narrative(r: pd.Series) -> str:
    superficie = f"{r['superficie_ha']:.1f}"
    ebitda_pct = f"{r['ebitda_margin']*100:.1f}%"
    deb_ebitda = f"{r['debito_su_ebitda']:.1f}x"
    sub_pct = f"{r['rapporto_sussidi']*100:.1f}%"
    ricavi_mln = r["ricavi_totali_eur"] / 1e6
    debito_mln = r["debito_finanziario_eur"] / 1e6

    fattori = []

    if r["ebitda_margin"] < 0.05:
        fattori.append("redditività operativa molto contenuta")
    elif r["ebitda_margin"] < 0.15:
        fattori.append("redditività operativa moderata")
    else:
        fattori.append("buona capacità di generare margini operativi")

    if r["debito_su_ebitda"] > 6:
        fattori.append("elevato livello di leva finanziaria")
    elif r["debito_su_ebitda"] > 3:
        fattori.append("indebitamento da monitorare")

    if r["rapporto_sussidi"] > 0.5:
        fattori.append("forte dipendenza dai pagamenti pubblici")
    elif r["rapporto_sussidi"] > 0.3:
        fattori.append("parziale dipendenza dai pagamenti pubblici")

    if r["score_tecnico_ambientale"] < 0.4:
        fattori.append("profilo tecnico-ambientale fragile")
    elif r["score_tecnico_ambientale"] > 0.7:
        fattori.append("buon allineamento ai requisiti tecnico-ambientali")

    if r["anni_inadempienze_ultimi5"] > 0:
        fattori.append("presenza di inadempienze pregresse nei rapporti con la PA")

    if fattori:
        fattori_txt = "; ".join(fattori)
    else:
        fattori_txt = "profilo complessivamente equilibrato, senza particolari elementi di criticità emersi"

    testo = (
        f"{r['denominazione']} è un'azienda agricola situata in {r['regione']}, "
        f"con circa {superficie} ettari dedicati prevalentemente a {r['coltura_principale'].lower()}. "
        f"I ricavi annui complessivi stimati ammontano a circa {ricavi_mln:.2f} milioni di euro, "
        f"a fronte di un indebitamento finanziario di circa {debito_mln:.2f} milioni. "
        f"Il profilo di rischio creditizio è classificato {r['classe_rating']} con uno score pari a "
        f"{r['score_rischio']:.1f} su 100. "
        f"La redditività operativa (EBITDA margin {ebitda_pct}) e il rapporto debito/EBITDA ({deb_ebitda}) "
        f"si combinano con un grado di dipendenza dai sussidi pubblici pari a {sub_pct} dei ricavi "
        f"e con un livello di conformità tecnico-ambientale valutato al "
        f"{r['score_tecnico_ambientale']*100:.1f}%. "
        f"In sintesi, i principali fattori di rischio economico-finanziario identificati sono: {fattori_txt}."
    )
    return testo


def compute_score_breakdown(r: pd.Series) -> dict:
    comp_ebitda = float(
        np.interp(r["ebitda_margin"], [-0.2, 0.0, 0.2, 0.4],
                  [0, 20, 30, 35]).clip(0, 35)
    )
    comp_debito = float(
        np.interp(r["debito_su_ebitda"], [0, 2, 4, 8, 12],
                  [30, 25, 20, 10, 0]).clip(0, 30)
    )
    comp_sussidi = float(
        np.interp(r["rapporto_sussidi"], [0.0, 0.2, 0.4, 0.7, 0.8],
                  [15, 15, 10, 5, 0]).clip(0, 15)
    )
    comp_tecnico = float(r["score_tecnico_ambientale"] * 15)
    comp_penale = float(r["penale_inadempienze"] * 25)

    totale = comp_ebitda + comp_debito + comp_sussidi + comp_tecnico - comp_penale
    totale_clipped = max(0.0, min(100.0, totale))

    return {
        "ebitda": comp_ebitda,
        "debito": comp_debito,
        "sussidi": comp_sussidi,
        "tecnico": comp_tecnico,
        "penale": comp_penale,
        "totale_raw": totale,
        "totale": totale_clipped,
    }


# ------------------------------------------------
# 4. Interfaccia Streamlit
# ------------------------------------------------

def main():
    st.set_page_config(page_title="AgriRating - A credit risk rating system for farms", layout="wide")

col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("logo_agrirating.png", width=120)

with col_title:
    st.title("AgriRating - A credit risk rating system for farms")
    st.caption("Demo con dati sintetici da fascicolo aziendale, rese, aiuti, eco-schemi, conformità.")

    if "data" not in st.session_state:
        base_df = generate_dummy_data()
        st.session_state["data"] = compute_rating(base_df)

    df = st.session_state["data"]

    # Sidebar filtri globali
    st.sidebar.header("Filtri portafoglio")

    regioni = ["Tutte"] + sorted(df["regione"].unique().tolist())
    regione_sel = st.sidebar.selectbox("Regione", regioni)

    colture = ["Tutte"] + sorted(df["coltura_principale"].unique().tolist())
    coltura_sel = st.sidebar.selectbox("Coltura principale", colture)

    classi = sorted(df["classe_rating"].unique().tolist())
    classi_sel = st.sidebar.multiselect(
        "Classi di rischio",
        options=classi,
        default=classi,
    )

    min_score = st.sidebar.slider(
        "Score minimo (0–100, alto = rischio basso)",
        0, 100, 0, 5
    )

    # Applicazione filtri
    df_filt = df.copy()
    if regione_sel != "Tutte":
        df_filt = df_filt[df_filt["regione"] == regione_sel]
    if coltura_sel != "Tutte":
        df_filt = df_filt[df_filt["coltura_principale"] == coltura_sel]
    df_filt = df_filt[df_filt["classe_rating"].isin(classi_sel)]
    df_filt = df_filt[df_filt["score_rischio"] >= min_score]

    # Tabs
    tab_port, tab_reg, tab_crop, tab_class, tab_farm, tab_analysis, tab_map = st.tabs([
        "Portafoglio",
        "Per regione",
        "Per coltura",
        "Per classe di rischio",
        "Scheda azienda",
        "Analisi dati",
        "Mappa rischio territoriale",
    ])

    # Portafoglio
    with tab_port:
        col1, col2, col3, col4 = st.columns(4)
        if len(df_filt) > 0:
            col1.metric("Numero aziende", len(df_filt))
            col2.metric("Score medio", f"{df_filt['score_rischio'].mean():.1f}")
            col3.metric("Ricavi totali (M€)", f"{df_filt['ricavi_totali_eur'].sum()/1e6:,.1f}")
            col4.metric("Debito totale (M€)", f"{df_filt['debito_finanziario_eur'].sum()/1e6:,.1f}")
        else:
            col1.metric("Numero aziende", 0)
            col2.metric("Score medio", "-")
            col3.metric("Ricavi totali (M€)", "-")
            col4.metric("Debito totale (M€)", "-")

        st.subheader("Distribuzione classi di rischio")
        if len(df_filt) > 0:
            rating_counts = df_filt["classe_rating"].value_counts().sort_index()
            st.bar_chart(rating_counts)
        else:
            st.info("Nessuna azienda dopo i filtri.")

        st.subheader("Elenco aziende")
        if len(df_filt) > 0:
            cols_show = [
                "farm_id", "denominazione", "regione", "coltura_principale",
                "superficie_ha", "score_rischio", "classe_rating",
                "ebitda_margin", "debito_su_ebitda", "rapporto_sussidi",
            ]
            st.dataframe(
                df_filt[cols_show].sort_values("score_rischio", ascending=False),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Nessuna azienda da visualizzare.")

    # Per regione
    with tab_reg:
        st.subheader("Sintesi per regione")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            agg_reg = df_filt.groupby("regione").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                ricavi_totali=("ricavi_totali_eur", "sum"),
                debito_totale=("debito_finanziario_eur", "sum"),
            ).sort_values("score_medio", ascending=False)
            st.dataframe(agg_reg, width="stretch")
            st.subheader("Score medio per regione")
            st.bar_chart(agg_reg["score_medio"])

    # Per coltura
    with tab_crop:
        st.subheader("Sintesi per coltura principale")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            agg_crop = df_filt.groupby("coltura_principale").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                ricavi_totali=("ricavi_totali_eur", "sum"),
            ).sort_values("score_medio", ascending=False)
            st.dataframe(agg_crop, width="stretch")
            st.subheader("Numero aziende per coltura")
            st.bar_chart(agg_crop["n_aziende"])

    # Per classe
    with tab_class:
        st.subheader("Distribuzione per classe di rischio")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            agg_class = df_filt.groupby("classe_rating").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                ricavi_totali=("ricavi_totali_eur", "sum"),
            ).sort_index()
            st.dataframe(agg_class, width="stretch")
            st.bar_chart(agg_class["n_aziende"])

    # Scheda azienda + motore di calcolo
    with tab_farm:
        st.subheader("Scheda creditizia sintetica")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            farm_opts = df_filt["farm_id"] + " – " + df_filt["denominazione"]
            selected = st.selectbox("Seleziona azienda", farm_opts)
            sel_id = selected.split(" – ")[0]
            row = df_filt[df_filt["farm_id"] == sel_id].iloc[0]

            ca, cb, cc = st.columns(3)
            ca.metric("Classe di rating", row["classe_rating"])
            ca.metric("Score di rischio", f"{row['score_rischio']:.1f}")
            cb.metric("EBITDA margin", f"{row['ebitda_margin']*100:.1f}%")
            cb.metric("Debito / EBITDA", f"{row['debito_su_ebitda']:.1f}x")
            cc.metric("Rapporto sussidi", f"{row['rapporto_sussidi']*100:.1f}%")
            cc.metric("Score tecnico-ambientale", f"{row['score_tecnico_ambientale']*100:.1f}%")

            st.markdown("**Descrizione sintetica (testo parametrico):**")
            st.write(build_company_narrative(row))

            st.markdown("**Motore di calcolo del rating (driver e fonti dati):**")
            breakdown = compute_score_breakdown(row)
            breakdown_df = pd.DataFrame([
                [
                    "Margine operativo (EBITDA margin)",
                    f"{row['ebitda_margin']*100:.1f}%",
                    "fino a 35 punti",
                    f"{breakdown['ebitda']:.1f}",
                    "Ricavi agricoli (rese × prezzi) + pagamenti PAC/eco-schemi da fascicolo aziendale / SIAN; costi standard per coltura",
                ],
                [
                    "Leva finanziaria (Debito / EBITDA)",
                    f"{row['debito_su_ebitda']:.1f}x",
                    "fino a 30 punti",
                    f"{breakdown['debito']:.1f}",
                    "Esposizione debitoria verso banche / confidi; EBITDA calcolato da dati tecnici e contabili",
                ],
                [
                    "Dipendenza da sussidi (rapporto sussidi/ricavi)",
                    f"{row['rapporto_sussidi']*100:.1f}%",
                    "fino a 15 punti",
                    f"{breakdown['sussidi']:.1f}",
                    "Pagamenti/aiuti PAC (titoli, eco-schemi, misure a superficie) da sistemi SIAN/AGEA; ricavi stimati da superfici, rese e prezzi medi",
                ],
                [
                    "Profilo tecnico-ambientale",
                    f"{row['score_tecnico_ambientale']*100:.1f}%",
                    "fino a 15 punti",
                    f"{breakdown['tecnico']:.1f}",
                    "Eco-schemi, condizionalità, controlli in loco/telematici, indicatori di conformità ambientale",
                ],
                [
                    "Penalità per inadempienze",
                    f"{row['anni_inadempienze_ultimi5']} anni su 5",
                    "-25 punti × penale",
                    f"-{breakdown['penale']:.1f}",
                    "Storico inadempienze amministrative: revoche, recuperi, sanzioni, irregolarità da archivi PA",
                ],
            ], columns=[
                "Driver",
                "Valore osservato",
                "Peso nel modello",
                "Punteggio parziale",
                "Fonte dati (attesa)",
            ])
            st.dataframe(breakdown_df, width="stretch", hide_index=True)

            st.markdown(
                f"**Score complessivo calcolato:** {breakdown['totale']:.1f} "
                f"(valore registrato: {row['score_rischio']:.1f})"
            )

            st.markdown("**Dati di base:**")
            info_cols = [
                "regione", "superficie_ha", "coltura_principale", "rese_t_ha",
                "ricavi_mercato_eur", "pagamenti_pubblici_eur",
                "costi_operativi_eur", "debito_finanziario_eur",
                "eco_schemi_score", "compliance_score",
                "anni_inadempienze_ultimi5",
            ]
            detail_df = (
                row[info_cols]
                .to_frame("Valore")
                .reset_index()
                .rename(columns={"index": "Indicatore"})
            )
            detail_df["Valore"] = detail_df["Valore"].astype(str)
            st.dataframe(detail_df, width="stretch", hide_index=True)

    # Analisi dati
    with tab_analysis:
        st.subheader("Analisi dati portafoglio")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            analysis_type = st.selectbox(
                "Tipo di analisi",
                [
                    "Matrice classi × colture (numero aziende)",
                    "Matrice classi × colture (score medio)",
                ],
            )

            if "numero aziende" in analysis_type:
                pivot = pd.pivot_table(
                    df_filt,
                    index="classe_rating",
                    columns="coltura_principale",
                    values="farm_id",
                    aggfunc="count",
                    fill_value=0,
                )
            else:
                pivot = pd.pivot_table(
                    df_filt,
                    index="classe_rating",
                    columns="coltura_principale",
                    values="score_rischio",
                    aggfunc="mean",
                    fill_value=0,
                )

            st.markdown("**Tabella dati:**")
            st.dataframe(pivot, width="stretch")

            st.markdown("**Heatmap:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig)

    # Mappa
    with tab_map:
        st.subheader("Mappa territoriale del rischio")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            df_map = df_filt.copy()
            df_map["color"] = df_map["classe_rating"].map(COLOR_MAP)

            tooltip = {
                "html": (
                    "<b>{denominazione}</b><br/>"
                    "Regione: {regione}<br/>"
                    "Coltura: {coltura_principale}<br/>"
                    "Classe: {classe_rating}<br/>"
                    "Score: {score_rischio}"
                ),
                "style": {"color": "white"},
            }

            view_state = pdk.ViewState(
                longitude=float(df_map["lon"].mean()),
                latitude=float(df_map["lat"].mean()),
                zoom=6,
                pitch=40,
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_radius=8000,
                get_fill_color="color",
                pickable=True,
            )

            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light",
            ))


if __name__ == "__main__":
    main()
