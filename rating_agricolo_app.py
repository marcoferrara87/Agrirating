import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# ------------------------------------------------
# PARAMETRI BASE / COLORI / CLASSI
# ------------------------------------------------

CROPS = ["Grano duro", "Mais", "Vite", "Olivo", "Ortofrutta", "Bovini latte"]

REGION_POINTS = {
    "Lazio": [
        (42.40, 12.85),
        (41.90, 12.60),
    ],
    "Campania": [
        (41.13, 14.78),
        (40.93, 14.80),
    ],
    "Puglia": [
        (41.47, 15.55),
        (40.83, 16.55),
    ],
    "Emilia-Romagna": [
        (44.70, 11.00),
        (44.90, 11.40),
    ],
    "Lombardia": [
        (45.60, 9.80),
        (45.50, 9.30),
    ],
    "Toscana": [
        (43.77, 11.25),
        (43.32, 11.33),
    ],
}

COLOR_MAP = {
    "A – Alta solvibilità": [0, 150, 0],
    "B – Solvibile": [100, 170, 0],
    "C – Vulnerabile": [200, 150, 0],
    "D – Rischiosa": [200, 80, 0],
    "E – Altamente rischiosa": [200, 0, 0],
}

# rating class → (PD_min, PD_max) in %
PD_MAP = {
    "A – Alta solvibilità": (0.0, 0.5),
    "B – Solvibile": (0.5, 1.5),
    "C – Vulnerabile": (1.5, 3.0),
    "D – Rischiosa": (3.0, 8.0),
    "E – Altamente rischiosa": (8.0, 20.0),
}

# ------------------------------------------------
# GENERAZIONE DATI FITTIZI
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

    # variabili strutturali / capitale
    df["valore_terreni_eur"] = (
        df["superficie_ha"] * np.random.uniform(15000, 40000, len(df))
    ).round(0)
    df["valore_fabbricati_eur"] = (
        df["superficie_ha"] * np.random.uniform(3000, 10000, len(df))
    ).round(0)
    df["indice_diversificazione"] = np.random.uniform(0.0, 1.0, len(df))

    # variabili andamentali / creditizie
    df["anni_rapporto_banca"] = np.random.randint(1, 21, len(df))
    df["numero_sconfinamenti_12m"] = np.random.randint(0, 6, len(df))
    df["giorni_medi_ritardo_pagamenti"] = np.random.randint(0, 61, len(df))
    df["cr_flag_sofferenza"] = np.random.binomial(1, 0.1, len(df))

    # coordinate interne, jitter ma ancorate a punti plausibili
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
# MOTORE ECONOMICO-FINANZIARIO DI BASE
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

    df["garanzie_reali_eur"] = (
        df["valore_terreni_eur"] * 0.7 + df["valore_fabbricati_eur"] * 0.5
    )
    df["loan_to_value"] = (
        df["debito_finanziario_eur"] / df["garanzie_reali_eur"].replace(0, np.nan)
    ).fillna(0).clip(0, 3)

    return df

# ------------------------------------------------
# DETTAGLIO PER MODULO (CALCOLO TRASPARENTE PER AZIENDA)
# ------------------------------------------------

def module_econ_detail(r: pd.Series) -> dict:
    comp_ebitda_norm = float(np.interp(
        r["ebitda_margin"], [-0.2, 0.0, 0.2, 0.4], [0.0, 0.3, 0.8, 1.0]
    ))
    comp_debito_norm = float(np.interp(
        r["debito_su_ebitda"], [0, 2, 4, 8, 12], [1.0, 0.8, 0.5, 0.2, 0.0]
    ))
    comp_sussidi_norm = float(np.interp(
        r["rapporto_sussidi"], [0.0, 0.2, 0.4, 0.7, 0.8], [0.8, 0.8, 0.5, 0.2, 0.0]
    ))
    econ_raw = 0.4 * comp_ebitda_norm + 0.4 * comp_debito_norm + 0.2 * comp_sussidi_norm
    econ_raw = max(0.0, min(1.0, econ_raw))
    score = econ_raw * 40.0
    return {
        "comp_ebitda_norm": comp_ebitda_norm,
        "comp_debito_norm": comp_debito_norm,
        "comp_sussidi_norm": comp_sussidi_norm,
        "econ_raw": econ_raw,
        "score": score,
        "contrib_ebitda": 0.4 * comp_ebitda_norm * 40.0,
        "contrib_debito": 0.4 * comp_debito_norm * 40.0,
        "contrib_sussidi": 0.2 * comp_sussidi_norm * 40.0,
    }

def module_and_detail(r: pd.Series) -> dict:
    sconfin = float(np.clip(r["numero_sconfinamenti_12m"], 0, 5))
    ritardi = float(np.clip(r["giorni_medi_ritardo_pagamenti"], 0, 60))
    soff = int(r["cr_flag_sofferenza"])

    sconfin_norm = float(np.interp(sconfin, [0, 1, 3, 5], [1.0, 0.8, 0.3, 0.0]))
    ritardi_norm = float(np.interp(ritardi, [0, 10, 30, 60], [1.0, 0.8, 0.4, 0.0]))
    soff_norm = 0.0 if soff == 1 else 1.0

    and_raw = 0.4 * sconfin_norm + 0.3 * ritardi_norm + 0.3 * soff_norm
    and_raw = max(0.0, min(1.0, and_raw))
    score = and_raw * 20.0

    return {
        "sconfin": sconfin,
        "ritardi": ritardi,
        "sofferenza_flag": soff,
        "sconfin_norm": sconfin_norm,
        "ritardi_norm": ritardi_norm,
        "soff_norm": soff_norm,
        "and_raw": and_raw,
        "score": score,
        "contrib_sconfin": 0.4 * sconfin_norm * 20.0,
        "contrib_ritardi": 0.3 * ritardi_norm * 20.0,
        "contrib_soff": 0.3 * soff_norm * 20.0,
    }

def module_strutt_detail(r: pd.Series) -> dict:
    superf = float(np.clip(r["superficie_ha"], 5, 250))
    superf_norm = float(np.interp(superf, [5, 30, 80, 250], [0.4, 1.0, 0.9, 0.5]))
    divers_norm = float(np.clip(r["indice_diversificazione"], 0, 1))
    strutt_raw = 0.6 * superf_norm + 0.4 * divers_norm
    strutt_raw = max(0.0, min(1.0, strutt_raw))
    score = strutt_raw * 10.0
    return {
        "superficie": superf,
        "superficie_norm": superf_norm,
        "diversificazione": divers_norm,
        "strutt_raw": strutt_raw,
        "score": score,
        "contrib_superficie": 0.6 * superf_norm * 10.0,
        "contrib_diversificazione": 0.4 * divers_norm * 10.0,
    }

def module_cap_detail(r: pd.Series) -> dict:
    ltv = float(np.clip(r["loan_to_value"], 0, 3))
    ltv_norm = float(np.interp(ltv, [0.0, 0.4, 0.7, 1.0, 3.0], [1.0, 1.0, 0.7, 0.3, 0.0]))
    anni_rel = float(np.clip(r["anni_rapporto_banca"], 1, 20))
    anni_rel_norm = float(np.interp(anni_rel, [1, 5, 10, 20], [0.3, 0.6, 0.9, 1.0]))
    cap_raw = 0.7 * ltv_norm + 0.3 * anni_rel_norm
    cap_raw = max(0.0, min(1.0, cap_raw))
    score = cap_raw * 10.0
    return {
        "ltv": ltv,
        "ltv_norm": ltv_norm,
        "anni_rapporto": anni_rel,
        "anni_rapporto_norm": anni_rel_norm,
        "cap_raw": cap_raw,
        "score": score,
        "contrib_ltv": 0.7 * ltv_norm * 10.0,
        "contrib_anni": 0.3 * anni_rel_norm * 10.0,
    }

def module_tec_detail(r: pd.Series) -> dict:
    tec_norm = float(np.clip(r["score_tecnico_ambientale"], 0, 1))
    score = tec_norm * 20.0
    return {
        "tec_norm": tec_norm,
        "score": score,
    }

# ------------------------------------------------
# CALCOLO COMPLESSIVO RATING (USA I MODULI SOPRA)
# ------------------------------------------------

def compute_modules_row(r: pd.Series) -> pd.Series:
    econ = module_econ_detail(r)
    andm = module_and_detail(r)
    strutt = module_strutt_detail(r)
    cap = module_cap_detail(r)
    tec = module_tec_detail(r)

    penale_pts = float(np.clip(r["penale_inadempienze"] * 15.0, 0, 30))

    score = econ["score"] + andm["score"] + strutt["score"] + cap["score"] + tec["score"] - penale_pts
    score = max(0.0, min(100.0, score))

    if score >= 85:
        rating_class = "A – Alta solvibilità"
    elif score >= 70:
        rating_class = "B – Solvibile"
    elif score >= 55:
        rating_class = "C – Vulnerabile"
    elif score >= 40:
        rating_class = "D – Rischiosa"
    else:
        rating_class = "E – Altamente rischiosa"

    pd_min, pd_max = PD_MAP[rating_class]
    pd_centrale = (pd_min + pd_max) / 2.0

    parts = []
    if econ["score"] < 20:
        parts.append("modulo economico-finanziario debole")
    elif econ["score"] > 30:
        parts.append("modulo economico-finanziario solido")

    if andm["score"] < 10:
        parts.append("profilo andamentale con elementi di attenzione")
    elif andm["score"] > 15:
        parts.append("buon track record andamentale")

    if strutt["score"] < 5:
        parts.append("struttura produttiva da rafforzare")
    else:
        parts.append("struttura produttiva adeguata")

    if cap["score"] < 5:
        parts.append("copertura tramite capitale fondiario limitata")
    else:
        parts.append("capitale fondiario/agrario a supporto della posizione creditizia")

    if tec["score"] < 10:
        parts.append("profilo tecnico-ambientale e di compliance con criticità")
    else:
        parts.append("profilo tecnico-ambientale e di compliance tendenzialmente positivo")

    spiegazione = "; ".join(parts) + "."

    return pd.Series({
        "score_mod_econ": round(econ["score"], 1),
        "score_mod_and": round(andm["score"], 1),
        "score_mod_strutt": round(strutt["score"], 1),
        "score_mod_cap": round(cap["score"], 1),
        "score_mod_tec": round(tec["score"], 1),
        "penale_pts": round(penale_pts, 1),
        "score_rischio": round(score, 1),
        "classe_rating": rating_class,
        "pd_min": pd_min,
        "pd_max": pd_max,
        "pd_centrale": pd_centrale,
        "spiegazione_rating": spiegazione,
    })

def compute_rating(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_financial_drivers(df)
    mods = df.apply(compute_modules_row, axis=1)
    for col in mods.columns:
        df[col] = mods[col]
    return df

# ------------------------------------------------
# NARRATIVA
# ------------------------------------------------

def build_company_narrative(r: pd.Series) -> str:
    superficie = f"{r['superficie_ha']:.1f}"
    ricavi_mln = r["ricavi_totali_eur"] / 1e6
    debito_mln = r["debito_finanziario_eur"] / 1e6
    pd_min = r["pd_min"]
    pd_max = r["pd_max"]

    testo = (
        f"{r['denominazione']} è un'azienda agricola situata in {r['regione']}, "
        f"con circa {superficie} ettari dedicati prevalentemente a {r['coltura_principale'].lower()}. "
        f"I ricavi annui complessivi stimati ammontano a circa {ricavi_mln:.2f} milioni di euro, "
        f"a fronte di un indebitamento finanziario di circa {debito_mln:.2f} milioni. "
        f"Il profilo di rischio creditizio è classificato {r['classe_rating']} "
        f"con uno score interno pari a {r['score_rischio']:.1f} su 100, "
        f"corrispondente a una Probability of Default stimata nell'intervallo "
        f"{pd_min:.1f}–{pd_max:.1f}% su base annua. "
        f"La valutazione integra cinque moduli: economico-finanziario, andamentale/comportamentale, "
        f"strutturale-produttivo, capitale fondiario/agrario e profilo tecnico-ambientale/compliance."
    )
    return testo

# ------------------------------------------------
# INTERFACCIA STREAMLIT
# ------------------------------------------------

def main():
    st.set_page_config(page_title="AgriRating - A model rating system by EY", layout="wide")

    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        # st.image("logo_agrirating.png", width=120)  # se hai un logo
        st.empty()
    with col_title:
        st.title("AgriRating - A model rating system by EY")
        st.caption(
            "POC di sistema di rating agricolo integrato (moduli economico-finanziario, andamentale, "
            "strutturale-produttivo, capitale fondiario/agrario, tecnico-ambientale/compliance)."
        )

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

    df_filt = df.copy()
    if regione_sel != "Tutte":
        df_filt = df_filt[df_filt["regione"] == regione_sel]
    if coltura_sel != "Tutte":
        df_filt = df_filt[df_filt["coltura_principale"] == coltura_sel]
    df_filt = df_filt[df_filt["classe_rating"].isin(classi_sel)]
    df_filt = df_filt[df_filt["score_rischio"] >= min_score]

    tab_port, tab_farm, tab_analysis, tab_map = st.tabs([
        "Portafoglio",
        "Scheda azienda",
        "Analisi dati",
        "Mappa rischio territoriale",
    ])

    # ----------------- PORTAFOGLIO -----------------
    with tab_port:
        st.subheader("Vista complessiva portafoglio (filtri applicati dalla sidebar)")
        col1, col2, col3, col4, col5 = st.columns(5)
        if len(df_filt) > 0:
            col1.metric("Numero aziende", len(df_filt))
            col2.metric("Score medio", f"{df_filt['score_rischio'].mean():.1f}")
            col3.metric("PD media stimata", f"{df_filt['pd_centrale'].mean():.2f}%")
            col4.metric("Ricavi totali (M€)", f"{df_filt['ricavi_totali_eur'].sum()/1e6:,.1f}")
            col5.metric("Debito totale (M€)", f"{df_filt['debito_finanziario_eur'].sum()/1e6:,.1f}")
        else:
            col1.metric("Numero aziende", 0)
            col2.metric("Score medio", "-")
            col3.metric("PD media stimata", "-")
            col4.metric("Ricavi totali (M€)", "-")
            col5.metric("Debito totale (M€)", "-")

        st.markdown("### Distribuzione per classe di rating (A–E)")
        if len(df_filt) > 0:
            agg_class = df_filt.groupby("classe_rating").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                pd_media=("pd_centrale", "mean"),
            ).sort_index()
            st.dataframe(agg_class, width="stretch")
            st.bar_chart(agg_class["n_aziende"])
        else:
            st.info("Nessuna azienda dopo i filtri.")

        st.markdown("### Sintesi per regione")
        if len(df_filt) > 0:
            agg_reg = df_filt.groupby("regione").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                pd_media=("pd_centrale", "mean"),
                ricavi_totali=("ricavi_totali_eur", "sum"),
                debito_totale=("debito_finanziario_eur", "sum"),
            ).sort_values("score_medio", ascending=False)
            st.dataframe(agg_reg, width="stretch")
        else:
            st.info("Nessuna azienda per la vista regionale.")

        st.markdown("### Sintesi per coltura principale")
        if len(df_filt) > 0:
            agg_crop = df_filt.groupby("coltura_principale").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                pd_media=("pd_centrale", "mean"),
                ricavi_totali=("ricavi_totali_eur", "sum"),
            ).sort_values("score_medio", ascending=False)
            st.dataframe(agg_crop, width="stretch")
        else:
            st.info("Nessuna azienda per la vista per coltura.")

        st.markdown("### Elenco aziende (dettaglio sintetico)")
        if len(df_filt) > 0:
            cols_show = [
                "farm_id", "denominazione", "regione", "coltura_principale",
                "superficie_ha", "score_rischio", "classe_rating",
                "pd_centrale", "ebitda_margin", "debito_su_ebitda",
                "rapporto_sussidi",
            ]
            st.dataframe(
                df_filt[cols_show].sort_values("score_rischio", ascending=False),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Nessuna azienda da visualizzare.")

    # -------------- SCHEDA AZIENDA + MODULI CLICCABILI --------------
    with tab_farm:
        st.subheader("Scheda creditizia sintetica (azienda singola)")
        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            farm_opts = df_filt["farm_id"] + " – " + df_filt["denominazione"]
            selected = st.selectbox("Seleziona azienda", farm_opts)
            sel_id = selected.split(" – ")[0]
            row = df_filt[df_filt["farm_id"] == sel_id].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Classe di rating", row["classe_rating"])
            c1.metric("Score interno (0–100)", f"{row['score_rischio']:.1f}")
            c2.metric("PD min", f"{row['pd_min']:.2f}%")
            c2.metric("PD max", f"{row['pd_max']:.2f}%")
            c3.metric("EBITDA margin", f"{row['ebitda_margin']*100:.1f}%")
            c3.metric("Debito / EBITDA", f"{row['debito_su_ebitda']:.1f}x")
            c4.metric("Rapporto sussidi", f"{row['rapporto_sussidi']*100:.1f}%")
            c4.metric("Score tecnico-ambientale", f"{row['score_tecnico_ambientale']*100:.1f}%")

            st.markdown("#### Descrizione sintetica (testo parametrico)")
            st.write(build_company_narrative(row))

            st.markdown("#### Moduli di rating – dettaglio calcolo (clicca sul modulo)")
            tab_me, tab_ma, tab_ms, tab_mc, tab_mt = st.tabs([
                "Modulo economico-finanziario",
                "Modulo andamentale/comportamentale",
                "Modulo strutturale-produttivo",
                "Modulo capitale fondiario/agrario",
                "Modulo tecnico-ambientale/compliance",
            ])

            econ = module_econ_detail(row)
            andm = module_and_detail(row)
            strutt = module_strutt_detail(row)
            cap = module_cap_detail(row)
            tec = module_tec_detail(row)

            with tab_me:
                st.markdown("**Modulo economico-finanziario (0–40 punti)**")
                df_me = pd.DataFrame([
                    ["EBITDA margin", f"{row['ebitda_margin']*100:.1f}%", econ["comp_ebitda_norm"], "40%"],
                    ["Debito / EBITDA", f"{row['debito_su_ebitda']:.1f}x", econ["comp_debito_norm"], "40%"],
                    ["Rapporto sussidi/ricavi", f"{row['rapporto_sussidi']*100:.1f}%", econ["comp_sussidi_norm"], "20%"],
                ], columns=["Variabile", "Valore osservato", "Valore normalizzato (0–1)", "Peso nel modulo"])
                st.dataframe(df_me, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo economico-finanziario: **{econ['score']:.1f} / 40**")

            with tab_ma:
                st.markdown("**Modulo andamentale/comportamentale (0–20 punti)**")
                df_ma = pd.DataFrame([
                    ["Numero sconfinamenti 12 mesi", andm["sconfin"], andm["sconfin_norm"], "40%"],
                    ["Giorni medi ritardo pagamenti", andm["ritardi"], andm["ritardi_norm"], "30%"],
                    ["Presenza sofferenza", "Sì" if andm["sofferenza_flag"] == 1 else "No", andm["soff_norm"], "30%"],
                ], columns=["Indicatore", "Valore osservato", "Valore normalizzato (0–1)", "Peso nel modulo"])
                st.dataframe(df_ma, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo andamentale/comportamentale: **{andm['score']:.1f} / 20**")

            with tab_ms:
                st.markdown("**Modulo strutturale-produttivo (0–10 punti)**")
                df_ms = pd.DataFrame([
                    ["Superficie aziendale (ha)", strutt["superficie"], strutt["superficie_norm"], "60%"],
                    ["Indice di diversificazione", f"{strutt['diversificazione']:.2f}", strutt["diversificazione"], "40%"],
                ], columns=["Indicatore", "Valore osservato", "Valore normalizzato (0–1)", "Peso nel modulo"])
                st.dataframe(df_ms, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo strutturale-produttivo: **{strutt['score']:.1f} / 10**")

            with tab_mc:
                st.markdown("**Modulo capitale fondiario/agrario (0–10 punti)**")
                df_mc = pd.DataFrame([
                    ["Loan-to-Value (debito/garanzie reali)", f"{cap['ltv']:.2f}", cap["ltv_norm"], "70%"],
                    ["Anni di rapporto con la banca", cap["anni_rapporto"], cap["anni_rapporto_norm"], "30%"],
                ], columns=["Indicatore", "Valore osservato", "Valore normalizzato (0–1)", "Peso nel modulo"])
                st.dataframe(df_mc, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo capitale fondiario/agrario: **{cap['score']:.1f} / 10**")

            with tab_mt:
                st.markdown("**Modulo tecnico-ambientale/compliance (0–20 punti)**")
                df_mt = pd.DataFrame([
                    ["Score tecnico-ambientale/compliance", f"{row['score_tecnico_ambientale']*100:.1f}%", tec["tec_norm"], "100%"],
                ], columns=["Indicatore", "Valore osservato", "Valore normalizzato (0–1)", "Peso nel modulo"])
                st.dataframe(df_mt, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo tecnico-ambientale/compliance: **{tec['score']:.1f} / 20**")

            st.markdown("#### Spiegazione sintetica del rating")
            st.write(row["spiegazione_rating"])

    # ----------------- ANALISI DATI -----------------
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

            st.markdown("**Tabella di base:**")
            st.dataframe(pivot, width="stretch")

            st.markdown("**Heatmap classi × colture:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig)

        st.markdown("### Modello dati target (concettuale)")
        st.write(
            "Il modello dati target per un rating agricolo regolamentare integra, oltre ai dati "
            "tecnico-produttivi (fascicolo aziendale, superfici, colture, rese, pagamenti PAC/eco-schemi), "
            "ulteriori famiglie di informazioni: Centrale Rischi, dati fiscali (Agenzia Entrate), "
            "bilanci e informazioni settoriali (CERVED o equivalenti), dati di filiera, open data climatici "
            "e remote sensing. Il POC simula alcuni di questi moduli attraverso variabili proxy, "
            "ma la logica di normalizzazione per coltura/area/cluster e l'utilizzo di orizzonti pluriennali "
            "andrebbero implementati in un data model esteso."
        )

    # ----------------- MAPPA -----------------
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
                    "Score: {score_rischio}<br/>"
                    "PD centrale: {pd_centrale:.2f}%"
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
