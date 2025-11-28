import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# Helper per compatibilità rerun con versioni diverse di Streamlit
def _rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()

# ------------------------------------------------
# PARAMETRI BASE / CLASSI / COLORI
# ------------------------------------------------

CROPS = ["Grano duro", "Mais", "Vite", "Olivo", "Ortofrutta", "Bovini latte"]

REGION_POINTS = {
    "Lazio": [(42.40, 12.85), (41.90, 12.60)],
    "Campania": [(41.13, 14.78), (40.93, 14.80)],
    "Puglia": [(41.47, 15.55), (40.83, 16.55)],
    "Emilia-Romagna": [(44.70, 11.00), (44.90, 11.40)],
    "Lombardia": [(45.60, 9.80), (45.50, 9.30)],
    "Toscana": [(43.77, 11.25), (43.32, 11.33)],
}

COLOR_MAP = {
    "A – Alta solvibilità": [0, 150, 0],
    "B – Solvibile": [100, 170, 0],
    "C – Vulnerabile": [200, 150, 0],
    "D – Rischiosa": [200, 80, 0],
    "E – Altamente rischiosa": [200, 0, 0],
}

# classe → intervallo di PD stimata (% annua)
PD_MAP = {
    "A – Alta solvibilità": (0.0, 0.5),
    "B – Solvibile": (0.5, 1.5),
    "C – Vulnerabile": (1.5, 3.0),
    "D – Rischiosa": (3.0, 8.0),
    "E – Altamente rischiosa": (8.0, 20.0),
}

# ------------------------------------------------
# 1. GENERAZIONE DATI FITTIZI (INCLUSA BASE PER SERIE STORICA)
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
        "eco_schemi_score": eco,          # n. eco-schemi attivati (0–5)
        "compliance_score": comp,         # indicatori di conformità (0–10)
        "anni_inadempienze_ultimi5": inademp,
    })

    df["prezzo_t"] = df["coltura_principale"].map(price_map)
    df["costo_ha"] = df["coltura_principale"].map(cost_ha_map)

    # RICAVI DA MERCATO = superficie × rese × prezzo medio
    df["ricavi_mercato_eur"] = (
        df["superficie_ha"] * df["rese_t_ha"] * df["prezzo_t"]
    ).round(0)

    # PAGAMENTI PAC/ECO-SCHEMI (titolo base + bonus eco-schemi)
    pac_ha_base = 250
    pac_ha_eco_bonus = 60
    df["pagamenti_pubblici_eur"] = (
        df["superficie_ha"] * (pac_ha_base + df["eco_schemi_score"] * pac_ha_eco_bonus)
    ).round(0)

    # COSTI OPERATIVI = superficie × costo/ha × fattore casuale
    df["costi_operativi_eur"] = (
        df["superficie_ha"] * df["costo_ha"] *
        np.random.uniform(0.9, 1.2, len(df))
    ).round(0)

    # DEBITO FINANZIARIO STIMATO
    df["debito_finanziario_eur"] = (
        df["ricavi_mercato_eur"] * np.random.uniform(0.2, 1.8, len(df))
    ).round(0)

    # STRUTTURA / CAPITALE
    df["valore_terreni_eur"] = (
        df["superficie_ha"] * np.random.uniform(15000, 40000, len(df))
    ).round(0)
    df["valore_fabbricati_eur"] = (
        df["superficie_ha"] * np.random.uniform(3000, 10000, len(df))
    ).round(0)
    df["indice_diversificazione"] = np.random.uniform(0.0, 1.0, len(df))

    # ANDAMENTALE / CREDITIZIO
    df["anni_rapporto_banca"] = np.random.randint(1, 21, len(df))
    df["numero_sconfinamenti_12m"] = np.random.randint(0, 6, len(df))
    df["giorni_medi_ritardo_pagamenti"] = np.random.randint(0, 61, len(df))
    df["cr_flag_sofferenza"] = np.random.binomial(1, 0.1, len(df))

    # COORDINATE
    lats, lons = [], []
    for reg in df["regione"]:
        base_lat, base_lon = REGION_POINTS[reg][np.random.randint(2)]
        lats.append(base_lat + np.random.normal(0, 0.08))
        lons.append(base_lon + np.random.normal(0, 0.08))
    df["lat"] = lats
    df["lon"] = lons

    # BASE PER SERIE STORICA: fattori di variazione casuali
    df["trend_ricavi"] = np.random.uniform(0.95, 1.05, len(df))
    df["trend_ebitda_margin"] = np.random.uniform(0.97, 1.03, len(df))
    df["trend_eco"] = np.random.randint(-1, 2, len(df))

    return df

# ------------------------------------------------
# 2. MOTORE FINANZIARIO DI BASE (FORMULE)
# ------------------------------------------------

def compute_financial_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RICAVI TOTALI = Ricavi di mercato + Pagamenti pubblici (PAC, eco-schemi, aiuti)
    df["ricavi_totali_eur"] = df["ricavi_mercato_eur"] + df["pagamenti_pubblici_eur"]

    # EBITDA = Ricavi totali – Costi operativi
    df["ebitda_eur"] = df["ricavi_totali_eur"] - df["costi_operativi_eur"]

    # EBITDA "aggiustato" per evitare divisioni per zero o valori estremi
    ebitda_floor = 1_000
    df["ebitda_adj_eur"] = df["ebitda_eur"].clip(lower=ebitda_floor)

    # EBITDA MARGIN = EBITDA / Ricavi totali
    df["ebitda_margin"] = (df["ebitda_eur"] / df["ricavi_totali_eur"]).clip(-0.5, 0.6)

    # RAPPORTO SUSSIDI = Pagamenti pubblici / Ricavi totali
    df["rapporto_sussidi"] = (
        df["pagamenti_pubblici_eur"] / df["ricavi_totali_eur"]
    ).clip(0, 0.8)

    # DEBITO / EBITDA
    df["debito_su_ebitda"] = (
        df["debito_finanziario_eur"] / df["ebitda_adj_eur"]
    ).clip(0, 12)

    # SCORE TECNICO-AMBIENTALE (0–1) = 0.5 * AGRI + 0.3 * ECO + 0.2 * COMPLIANCE
    agri_proxy = df["eco_schemi_score"] / 5.0
    eco_norm = df["eco_schemi_score"] / 5.0
    comp_norm = df["compliance_score"] / 10.0
    df["score_tecnico_ambientale"] = (
        0.5 * agri_proxy + 0.3 * eco_norm + 0.2 * comp_norm
    ).clip(0, 1)

    # PENALITÀ INADEMPIENZE (ultimi 5 anni)
    df["penale_inadempienze"] = df["anni_inadempienze_ultimi5"] * 0.1

    # GARANZIE REALI = 70% valore terreni + 50% fabbricati
    df["garanzie_reali_eur"] = (
        df["valore_terreni_eur"] * 0.7 + df["valore_fabbricati_eur"] * 0.5
    )

    # LOAN-TO-VALUE (LTV) = Debito finanziario / Garanzie reali
    df["loan_to_value"] = (
        df["debito_finanziario_eur"] / df["garanzie_reali_eur"].replace(0, np.nan)
    ).fillna(0).clip(0, 3)

    return df

# ------------------------------------------------
# 3. MODULI DI RATING – DETTAGLIO FORMULE PER AZIENDA
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

    # SCORE MODULO ECONOMICO-FINANZIARIO (0–40)
    econ_raw = 0.4 * comp_ebitda_norm + 0.4 * comp_debito_norm + 0.2 * comp_sussidi_norm
    econ_raw = max(0.0, min(1.0, econ_raw))
    score = econ_raw * 40.0

    return {
        "comp_ebitda_norm": comp_ebitda_norm,
        "comp_debito_norm": comp_debito_norm,
        "comp_sussidi_norm": comp_sussidi_norm,
        "econ_raw": econ_raw,
        "score": score,
    }

def module_and_detail(r: pd.Series) -> dict:
    sconfin = float(np.clip(r["numero_sconfinamenti_12m"], 0, 5))
    ritardi = float(np.clip(r["giorni_medi_ritardo_pagamenti"], 0, 60))
    soff = int(r["cr_flag_sofferenza"])

    # Normalizzazioni
    sconfin_norm = float(np.interp(sconfin, [0, 1, 3, 5], [1.0, 0.8, 0.3, 0.0]))
    ritardi_norm = float(np.interp(ritardi, [0, 10, 30, 60], [1.0, 0.8, 0.4, 0.0]))
    soff_norm = 0.0 if soff == 1 else 1.0

    # SCORE MODULO ANDAMENTALE (0–20)
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
    }

def module_strutt_detail(r: pd.Series) -> dict:
    superf = float(np.clip(r["superficie_ha"], 5, 250))
    superf_norm = float(np.interp(superf, [5, 30, 80, 250], [0.4, 1.0, 0.9, 0.5]))
    divers_norm = float(np.clip(r["indice_diversificazione"], 0, 1))

    # SCORE MODULO STRUTTURALE (0–10)
    strutt_raw = 0.6 * superf_norm + 0.4 * divers_norm
    strutt_raw = max(0.0, min(1.0, strutt_raw))
    score = strutt_raw * 10.0

    return {
        "superficie": superf,
        "superficie_norm": superf_norm,
        "diversificazione": divers_norm,
        "strutt_raw": strutt_raw,
        "score": score,
    }

def module_cap_detail(r: pd.Series) -> dict:
    ltv = float(np.clip(r["loan_to_value"], 0, 3))
    ltv_norm = float(np.interp(ltv, [0.0, 0.4, 0.7, 1.0, 3.0], [1.0, 1.0, 0.7, 0.3, 0.0]))
    anni_rel = float(np.clip(r["anni_rapporto_banca"], 1, 20))
    anni_rel_norm = float(np.interp(anni_rel, [1, 5, 10, 20], [0.3, 0.6, 0.9, 1.0]))

    # SCORE MODULO CAPITALE (0–10)
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
    }

def module_tec_detail(r: pd.Series) -> dict:
    tec_norm = float(np.clip(r["score_tecnico_ambientale"], 0, 1))
    # SCORE MODULO TECNICO-AMBIENTALE (0–20)
    score = tec_norm * 20.0
    return {"tec_norm": tec_norm, "score": score}

# ------------------------------------------------
# 4. CALCOLO COMPLESSIVO RATING (A–E + PD)
# ------------------------------------------------

def compute_modules_row(r: pd.Series) -> pd.Series:
    econ = module_econ_detail(r)
    andm = module_and_detail(r)
    strutt = module_strutt_detail(r)
    cap = module_cap_detail(r)
    tec = module_tec_detail(r)

    # PENALITÀ INADEMPIENZE (0–30 punti)
    penale_pts = float(np.clip(r["penale_inadempienze"] * 15.0, 0, 30))

    # SCORE COMPLESSIVO (0–100)
    score = econ["score"] + andm["score"] + strutt["score"] + cap["score"] + tec["score"] - penale_pts
    score = max(0.0, min(100.0, score))

    # MAPPATURA IN CLASSI A–E
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

    # SPIEGAZIONE TESTUALE
    parts = []
    if econ["score"] < 20:
        parts.append("modulo economico-finanziario debole (margini o leva da riequilibrare)")
    elif econ["score"] > 30:
        parts.append("modulo economico-finanziario complessivamente solido")

    if andm["score"] < 10:
        parts.append("profilo andamentale con elementi di attenzione (sconfinamenti/ritardi)")
    elif andm["score"] > 15:
        parts.append("storico andamentale regolare")

    if strutt["score"] < 5:
        parts.append("struttura produttiva di scala limitata o poco diversificata")
    else:
        parts.append("struttura produttiva adeguata al profilo di rischio")

    if cap["score"] < 5:
        parts.append("copertura tramite capitale fondiario/agrario limitata")
    else:
        parts.append("capitale fondiario/agrario a supporto rilevante")

    if tec["score"] < 10:
        parts.append("profilo tecnico-ambientale/compliance con alcune criticità")
    else:
        parts.append("profilo tecnico-ambientale/compliance tendenzialmente positivo")

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
# 5. NARRATIVA AZIENDALE
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
        f"e una Probabilità di Default (PD) stimata nell'intervallo {pd_min:.1f}–{pd_max:.1f}% su base annua. "
        f"La valutazione integra cinque moduli: economico-finanziario, andamentale/comportamentale, "
        f"strutturale-produttivo, capitale fondiario/agrario e tecnico-ambientale/compliance, "
        f"alimentati da dati amministrativi (fascicolo aziendale, PAC), informazioni creditizie simulate "
        f"e proxy di struttura e capitale fondiario."
    )
    return testo

# ------------------------------------------------
# 6. SERIE STORICA 5 ANNI PER AZIENDA (SIMULATA)
# ------------------------------------------------

def build_time_series(r: pd.Series) -> pd.DataFrame:
    """
    Crea una serie storica fittizia a 5 anni (t-4,...,t),
    partendo dai valori correnti e applicando i trend.
    """
    anni = [r"t-4", r"t-3", r"t-2", r"t-1", r"t"]
    ricavi = []
    ebitda_margins = []
    rese = []
    eco_scores = []
    inademp = []

    ricavi_base = r["ricavi_totali_eur"]
    ebitda_base = r["ebitda_margin"]
    rese_base = r["rese_t_ha"]
    eco_base = r["eco_schemi_score"]
    inademp_base = r["anni_inadempienze_ultimi5"]

    for i in range(5):
        # ricavi con trend e rumore
        fattore_ricavi = r["trend_ricavi"] ** (i - 4)
        ricavi.append(ricavi_base * fattore_ricavi * np.random.uniform(0.95, 1.05))
        # EBITDA margin
        fatt_ebitda = r["trend_ebitda_margin"] ** (i - 4)
        ebitda_margins.append(
            float(np.clip(ebitda_base * fatt_ebitda * np.random.uniform(0.95, 1.05), -0.5, 0.6))
        )
        # rese
        rese.append(
            float(np.clip(rese_base * np.random.uniform(0.9, 1.1), 1.5, 13.0))
        )
        # eco-schemi score
        eco_scores.append(
            int(np.clip(eco_base + r["trend_eco"] * (i - 4) + np.random.randint(-1, 2), 0, 5))
        )
        # inadempienze cumulative (proxy)
        inademp.append(
            max(0, inademp_base + np.random.randint(-1, 2))
        )

    hist_df = pd.DataFrame({
        "Anno": anni,
        "Ricavi totali stimati (EUR)": np.round(ricavi, 0),
        "EBITDA margin": ebitda_margins,
        "Rese t/ha": np.round(rese, 1),
        "Eco-schemi attivati": eco_scores,
        "Anni con inadempienze (cumulato)": inademp,
    })
    return hist_df

# ------------------------------------------------
# 7. LOGICA ASSISTENTE AI (SEMPLIFICATA)
# ------------------------------------------------

def ai_answer(query: str, df: pd.DataFrame) -> str:
    q = query.lower().strip()

    # Cerca un FARM-XXXX dentro alla domanda
    farm_id = None
    for fid in df["farm_id"].tolist():
        if fid.lower() in q:
            farm_id = fid
            break

    if farm_id is not None:
        r = df[df["farm_id"] == farm_id].iloc[0]
        return (
            f"Ho trovato **{r['denominazione']}** (`{r['farm_id']}`) in {r['regione']}.\n\n"
            f"- Classe di rating: **{r['classe_rating']}** (score {r['score_rischio']:.1f}/100)\n"
            f"- PD stimata: **{r['pd_min']:.2f}–{r['pd_max']:.2f}%**\n"
            f"- EBITDA margin: **{r['ebitda_margin']*100:.1f}%**\n"
            f"- Debito/EBITDA: **{r['debito_su_ebitda']:.1f}x**\n\n"
            f"{r['spiegazione_rating']}"
        )

    if "classe" in q and "rating" in q:
        return (
            "La classe di rating deriva da uno score interno 0–100.\n\n"
            "- Sommo i 5 moduli: economico-finanziario (0–40), andamentale (0–20), "
            "strutturale (0–10), capitale fondiario/agrario (0–10), tecnico-ambientale (0–20).\n"
            "- Tolgo fino a 30 punti di penalità per le inadempienze degli ultimi 5 anni.\n"
            "- Applico la scala: ≥85 = A, 70–84 = B, 55–69 = C, 40–54 = D, <40 = E.\n"
            "A ogni classe associo un intervallo di PD annua predefinito."
        )

    if "pd" in q or "probabilità di default" in q:
        return (
            "La PD non è calcolata in modo continuo ma agganciata alla classe di rating:\n\n"
            "- A – Alta solvibilità: 0,0–0,5% annua\n"
            "- B – Solvibile: 0,5–1,5%\n"
            "- C – Vulnerabile: 1,5–3,0%\n"
            "- D – Rischiosa: 3–8%\n"
            "- E – Altamente rischiosa: 8–20%\n\n"
            "Per il portafoglio uso il valore centrale dell’intervallo come PD puntuale."
        )

    if "rapporto sussidi" in q or "sussidi" in q:
        return (
            "Il rapporto sussidi è dato da **pagamenti pubblici / ricavi totali**.\n\n"
            "Misura quanto il fatturato aziendale dipende da PAC ed eco-schemi; "
            "più è alto, più il modello di business è esposto al rischio regolatorio "
            "sul sostegno pubblico."
        )

    return (
        "Posso spiegare formule dei moduli, intervalli di PD o sintetizzare il rating di una singola "
        "azienda (es. `dimmi qualcosa su FARM-0007`). Digita un ID azienda o fammi una domanda sui moduli."
    )

def render_ai_assistant(df: pd.DataFrame):
    if "ai_open" not in st.session_state:
        st.session_state["ai_open"] = False
    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Ciao, sono l'assistente AI di AgriRating. Posso spiegare le formule, la logica delle "
                    "classi A–E e riassumere il rating di una singola azienda. Prova con `FARM-0001`."
                ),
            }
        ]

    # CSS per fissare il blocco in basso a sinistra, sopra tutto
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(> #ai-chat-root) {
            position: fixed;
            left: 1.2rem;
            bottom: 1.2rem;
            max-width: 360px;
            width: 360px;
            z-index: 9999;
        }
        .ai-card {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.18);
            border: 1px solid rgba(15,23,42,0.08);
            overflow: hidden;
            font-size: 0.85rem;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .ai-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.45rem 0.7rem;
            background: linear-gradient(135deg, #00694b, #00a884);
            color: #ffffff;
        }
        .ai-header-left {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .ai-avatar {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #ffffff22;
            border: 1px solid #ffffff66;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .ai-title {
            font-size: 0.8rem;
            font-weight: 600;
            line-height: 1.1;
        }
        .ai-status {
            font-size: 0.7rem;
            opacity: 0.92;
        }
        .ai-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #4ade80;
            margin-right: 4px;
            box-shadow: 0 0 0 4px rgba(74,222,128,0.25);
        }
        .ai-body {
            padding: 0.45rem 0.6rem 0.3rem 0.6rem;
            background: #f9fafb;
            max-height: 260px;
            overflow-y: auto;
        }
        .ai-minimized-button button {
            border-radius: 999px !important;
            padding: 0.35rem 0.7rem !important;
            font-size: 0.8rem !important;
        }
        @media (max-width: 768px) {
            div[data-testid="stVerticalBlock"]:has(> #ai-chat-root) {
                left: 0.6rem;
                bottom: 0.6rem;
                width: calc(100vw - 1.2rem);
                max-width: calc(100vw - 1.2rem);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div id="ai-chat-root"></div>', unsafe_allow_html=True)

        # Stato: minimizzato
        if not st.session_state["ai_open"]:
            with st.container():
                st.markdown(
                    '<div class="ai-minimized-button">',
                    unsafe_allow_html=True,
                )
                if st.button("🤖 AgriRating AI", key="ai_open_btn"):
                    st.session_state["ai_open"] = True
                    _rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            return

        # Stato: pannello aperto
        st.markdown(
            """
            <div class="ai-card">
              <div class="ai-header">
                <div class="ai-header-left">
                  <div class="ai-avatar">AI</div>
                  <div>
                    <div class="ai-title">Assistente AgriRating</div>
                    <div class="ai-status"><span class="ai-dot"></span>Spiega moduli, PD e rating</div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Corpo chat
        st.markdown('<div class="ai-body">', unsafe_allow_html=True)
        for msg in st.session_state["ai_messages"]:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Input utente
        user_input = st.chat_input("Scrivi una domanda o un ID azienda (es. FARM-0007)...")

        if user_input:
            st.session_state["ai_messages"].append(
                {"role": "user", "content": user_input}
            )
            answer = ai_answer(user_input, df)
            st.session_state["ai_messages"].append(
                {"role": "assistant", "content": answer}
            )
            _rerun()

        # Pulsante chiusura
        close_col, _ = st.columns([1, 3])
        with close_col:
            if st.button("Chiudi", key="ai_close_btn"):
                st.session_state["ai_open"] = False
                _rerun()

# ------------------------------------------------
# 8. INTERFACCIA STREAMLIT
# ------------------------------------------------

def main():
    st.set_page_config(page_title="AgriRating - A model rating system by EY", layout="wide")

    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.empty()
    with col_title:
        st.title("AgriRating - A model rating system by EY")
        st.caption(
            "Prototipo di sistema di rating agricolo multi-modulo con classi A–E e Probabilità di Default (PD) stimata."
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
        "Classi di rischio (A–E)",
        options=classi,
        default=classi,
    )

    min_score = st.sidebar.slider(
        "Score interno minimo (0–100, alto = rischio basso)",
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
        st.subheader("Vista portafoglio (filtri applicati)")

        col1, col2, col3, col4, col5 = st.columns(5)
        if len(df_filt) > 0:
            col1.metric("Numero aziende in portafoglio", len(df_filt))
            col2.metric("Score interno medio", f"{df_filt['score_rischio'].mean():.1f}")
            col3.metric("PD media stimata", f"{df_filt['pd_centrale'].mean():.2f}%")
            col4.metric("Ricavi totali stimati (M€)", f"{df_filt['ricavi_totali_eur'].sum()/1e6:,.1f}")
            col5.metric("Debito finanziario totale (M€)", f"{df_filt['debito_finanziario_eur'].sum()/1e6:,.1f}")
        else:
            col1.metric("Numero aziende in portafoglio", 0)
            col2.metric("Score interno medio", "-")
            col3.metric("PD media stimata", "-")
            col4.metric("Ricavi totali stimati (M€)", "-")
            col5.metric("Debito finanziario totale (M€)", "-")

        st.markdown("### Distribuzione per classe di rating (A–E)")
        if len(df_filt) > 0:
            agg_class = df_filt.groupby("classe_rating").agg(
                n_aziende=("farm_id", "count"),
                score_medio=("score_rischio", "mean"),
                pd_media=("pd_centrale", "mean"),
            ).sort_index()
            st.dataframe(agg_class, width="stretch")
        else:
            st.info("Nessuna azienda dopo i filtri.")

        st.markdown("### Elenco aziende (vista sintetica)")
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

    # -------------- SCHEDA AZIENDA (MODULI CLICCABILI + FORMULE + FONTI) --------------
    with tab_farm:
        st.subheader("Scheda azienda – rating e moduli di calcolo")

        if len(df_filt) == 0:
            st.info("Nessuna azienda dopo i filtri.")
        else:
            farm_opts = df_filt["farm_id"] + " – " + df_filt["denominazione"]
            selected = st.selectbox("Seleziona azienda", farm_opts)
            sel_id = selected.split(" – ")[0]
            row = df_filt[df_filt["farm_id"] == sel_id].iloc[0]

            econ = module_econ_detail(row)
            andm = module_and_detail(row)
            strutt = module_strutt_detail(row)
            cap = module_cap_detail(row)
            tec = module_tec_detail(row)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Classe di rating", row["classe_rating"])
            c1.metric("Score interno (0–100)", f"{row['score_rischio']:.1f}")
            c2.metric("Probabilità di Default (PD) minima", f"{row['pd_min']:.2f}%")
            c2.metric("Probabilità di Default (PD) massima", f"{row['pd_max']:.2f}%")
            c3.metric("Margine operativo lordo (EBITDA margin)", f"{row['ebitda_margin']*100:.1f}%")
            c3.metric("Rapporto debito / EBITDA", f"{row['debito_su_ebitda']:.1f}x")
            c4.metric("Incidenza pagamenti PAC/eco-schemi", f"{row['rapporto_sussidi']*100:.1f}%")
            c4.metric("Indice tecnico-ambientale/compliance", f"{row['score_tecnico_ambientale']*100:.1f}%")

            st.markdown("#### Descrizione sintetica")
            st.write(build_company_narrative(row))

            st.markdown("#### Moduli di rating – clicca per vedere formule e fonti dati")
            tab_me, tab_ma, tab_ms, tab_mc, tab_mt, tab_hist = st.tabs([
                "Economico-finanziario",
                "Andamentale/comportamentale",
                "Strutturale-produttivo",
                "Capitale fondiario/agrario",
                "Tecnico-ambientale/compliance",
                "Storico 5 anni",
            ])

            # ECONOMICO-FINANZIARIO
            with tab_me:
                st.markdown("**Modulo economico-finanziario (0–40 punti)**")
                st.markdown(
                    """
                    Formule principali:

                    - Ricavi totali = Ricavi di mercato + Pagamenti pubblici (PAC, eco-schemi, aiuti)
                    - EBITDA = Ricavi totali − Costi operativi
                    - EBITDA margin = EBITDA / Ricavi totali
                    - Rapporto sussidi = Pagamenti pubblici / Ricavi totali
                    - Debito / EBITDA = Debito finanziario / EBITDA\\_aggiustato
                    - Score economico-finanziario = 40 × [0,4×EBITDA_norm + 0,4×Debito_norm + 0,2×Sussidi_norm]
                    """
                )

                df_me = pd.DataFrame([
                    [
                        "EBITDA margin",
                        f"{row['ebitda_margin']*100:.1f}%",
                        f"{econ['comp_ebitda_norm']:.2f}",
                        "0,4",
                        "Bilanci/dichiarazioni fiscali (Agenzia Entrate, CERVED), fascicolo aziendale",
                    ],
                    [
                        "Debito / EBITDA",
                        f"{row['debito_su_ebitda']:.1f}x",
                        f"{econ['comp_debito_norm']:.2f}",
                        "0,4",
                        "Informazioni creditizie (Centrale Rischi), bilancio, banca partner",
                    ],
                    [
                        "Rapporto sussidi / ricavi",
                        f"{row['rapporto_sussidi']*100:.1f}%",
                        f"{econ['comp_sussidi_norm']:.2f}",
                        "0,2",
                        "SIAN/AGEA (pagamenti PAC, eco-schemi), fascicolo aziendale",
                    ],
                ], columns=[
                    "Indicatore",
                    "Valore osservato",
                    "Valore normalizzato (0–1)",
                    "Peso nel modulo",
                    "Fonti dati previste",
                ])
                st.dataframe(df_me, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo economico-finanziario: **{econ['score']:.1f} / 40**")

            # ANDAMENTALE
            with tab_ma:
                st.markdown("**Modulo andamentale/comportamentale (0–20 punti)**")
                st.markdown(
                    """
                    Formule principali:

                    - Score andamentale = 20 × [0,4×Sconfinamenti_norm + 0,3×Ritardi_norm + 0,3×Sofferenze_norm]
                    - Sconfinamenti_norm: da 1,0 (nessuno) a 0,0 (molti sconfinamenti)
                    - Ritardi_norm: da 1,0 (nessun ritardo) a 0,0 (ritardi > 60 giorni)
                    - Sofferenze_norm: 1,0 se nessuna sofferenza; 0,0 se presente sofferenza
                    """
                )
                df_ma = pd.DataFrame([
                    [
                        "Numero sconfinamenti negli ultimi 12 mesi",
                        andm["sconfin"],
                        f"{andm['sconfin_norm']:.2f}",
                        "0,4",
                        "Centrale Rischi, sistemi informativi bancari",
                    ],
                    [
                        "Giorni medi di ritardo nei pagamenti",
                        andm["ritardi"],
                        f"{andm['ritardi_norm']:.2f}",
                        "0,3",
                        "Centrale Rischi, contabilità clienti/fornitori",
                    ],
                    [
                        "Presenza di posizioni a sofferenza",
                        "Sì" if andm["sofferenza_flag"] == 1 else "No",
                        f"{andm['soff_norm']:.2f}",
                        "0,3",
                        "Centrale Rischi, segnalazioni bancarie",
                    ],
                ], columns=[
                    "Indicatore",
                    "Valore osservato",
                    "Valore normalizzato (0–1)",
                    "Peso nel modulo",
                    "Fonti dati previste",
                ])
                st.dataframe(df_ma, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo andamentale/comportamentale: **{andm['score']:.1f} / 20**")

            # STRUTTURALE
            with tab_ms:
                st.markdown("**Modulo strutturale-produttivo (0–10 punti)**")
                st.markdown(
                    """
                    Formule principali:

                    - Superficie_norm: funzione della SAU aziendale (5–250 ha)
                    - Diversificazione_norm: indice 0–1 costruito su numero di produzioni, filiere, attività complementari
                    - Score strutturale = 10 × [0,6×Superficie_norm + 0,4×Diversificazione_norm]
                    """
                )
                df_ms = pd.DataFrame([
                    [
                        "Superficie agricola utilizzata (SAU, ha)",
                        strutt["superficie"],
                        f"{strutt['superficie_norm']:.2f}",
                        "0,6",
                        "Fascicolo aziendale SIAN, catasto",
                    ],
                    [
                        "Indice di diversificazione produttiva (0–1)",
                        f"{strutt['diversificazione']:.2f}",
                        f"{strutt['diversificazione']:.2f}",
                        "0,4",
                        "Fascicolo aziendale, anagrafe aziende, dati di filiera",
                    ],
                ], columns=[
                    "Indicatore",
                    "Valore osservato",
                    "Valore normalizzato (0–1)",
                    "Peso nel modulo",
                    "Fonti dati previste",
                ])
                st.dataframe(df_ms, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo strutturale-produttivo: **{strutt['score']:.1f} / 10**")

            # CAPITALE FONDIARIO
            with tab_mc:
                st.markdown("**Modulo capitale fondiario/agrario (0–10 punti)**")
                st.markdown(
                    """
                    Formule principali:

                    - Garanzie reali = 0,7×Valore terreni + 0,5×Valore fabbricati
                    - Loan-to-Value (LTV) = Debito finanziario / Garanzie reali
                    - Score capitale = 10 × [0,7×LTV_norm + 0,3×AnniRapporto_norm]
                    """
                )
                df_mc = pd.DataFrame([
                    [
                        "Loan-to-Value (debito / garanzie reali)",
                        f"{cap['ltv']:.2f}",
                        f"{cap['ltv_norm']:.2f}",
                        "0,7",
                        "Catasto, perizie fondiarie, Centrale Rischi, bilanci",
                    ],
                    [
                        "Anni di rapporto continuativo con la banca",
                        cap["anni_rapporto"],
                        f"{cap['anni_rapporto_norm']:.2f}",
                        "0,3",
                        "Sistemi bancari interni, Centrale Rischi",
                    ],
                ], columns=[
                    "Indicatore",
                    "Valore osservato",
                    "Valore normalizzato (0–1)",
                    "Peso nel modulo",
                    "Fonti dati previste",
                ])
                st.dataframe(df_mc, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo capitale fondiario/agrario: **{cap['score']:.1f} / 10**")

            # TECNICO-AMBIENTALE
            with tab_mt:
                st.markdown("**Modulo tecnico-ambientale/compliance (0–20 punti)**")
                st.markdown(
                    """
                    Formule principali (indicatore sintetico 0–1):

                    - Agri\\_proxy = EcoSchemi / 5
                    - Eco = EcoSchemi / 5
                    - Compliance = Compliance\\_score / 10
                    - Score tecnico-ambientale = 0,5×Agri_proxy + 0,3×Eco + 0,2×Compliance
                    - Score modulo tecnico-ambientale = 20 × Score tecnico-ambientale
                    """
                )
                df_mt = pd.DataFrame([
                    [
                        "Numero eco-schemi/ pratiche agro-ambientali attivate",
                        row["eco_schemi_score"],
                        f"{tec['tec_norm']:.2f}",
                        "0,5 (Agri) + 0,3 (Eco)",
                        "SIAN/AGEA, fascicolo aziendale, registri eco-schemi",
                    ],
                    [
                        "Indice di conformità e controlli (0–10)",
                        row["compliance_score"],
                        f"{row['compliance_score']/10.0:.2f}",
                        "0,2",
                        "SIAN, sistemi di controllo PAC, ispezioni, open data ambientali",
                    ],
                ], columns=[
                    "Indicatore",
                    "Valore osservato",
                    "Valore normalizzato (0–1)",
                    "Peso nel modulo",
                    "Fonti dati previste",
                ])
                st.dataframe(df_mt, width="stretch", hide_index=True)
                st.markdown(f"Punteggio modulo tecnico-ambientale/compliance: **{tec['score']:.1f} / 20**")

            # STORICO 5 ANNI
            with tab_hist:
                st.markdown("**Dimensione storica (ultimi 5 anni)** – dati simulati su base POC")
                hist_df = build_time_series(row)
                st.dataframe(hist_df, width="stretch", hide_index=True)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(hist_df["Anno"], hist_df["EBITDA margin"], marker="o")
                ax.set_ylabel("EBITDA margin")
                ax.set_xlabel("Anno (t = anno corrente)")
                st.pyplot(fig)

            st.markdown("#### Spiegazione sintetica del rating complessivo")
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
            "Il modello dati target per un rating agricolo regolamentare integrerebbe, oltre ai dati "
            "tecnico-produttivi (fascicolo aziendale, superfici, colture, rese, pagamenti PAC/eco-schemi), "
            "informazioni creditizie di Centrale Rischi, bilanci e dati fiscali (Agenzia Entrate, CERVED), "
            "dati di filiera, open data climatici e remote sensing, con normalizzazione per coltura/area "
            "e orizzonte storico pluriennale."
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
                    "Score interno: {score_rischio}<br/>"
                    "PD centrale stimata: {pd_centrale:.2f}%"
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

    # ----------------- AGENTE AI FLOTTE BOTTOM-LEFT -----------------
    render_ai_assistant(df)


if __name__ == "__main__":
    main()



