# streamlit_abba.py
# -------------------------------------------------------------
# Streamlit-Dashboard voor de berekening en visualisatie van debieten van de gemeente Almere volgens de ABBA-berekening. 
# Ruwe reeksen worden opgehaald uit de API van flevoland.lizard.net en omgerekend naar statusreeksen en debietreeksen (debiet per pomp en instroom) debiet.
# 
# -------------------------------------------------------------
# Volledig opgeschoond:
# - Eén set plot-functies + één renderblok (gedeelde zoom/pan)
# - Beide pompen gebruiken bit 9 (aanpasbaar in UI)
# - Status (bedrijf) vectorized 0/1, visueel ×1000
# - Pompdebieten = status × netto debiet, maar alléén bij uitstroom (debiet < 0)
#   en in de grafiek positief getekend (tekenflip)
# - debiet_in correct (zoals v7.2), los van bit-index
# - precip_cum uit raster bij LT5 (indien beschikbaar)
# - Clipping: alleen de debiet-lijn; pomppunten NIET meeclippen
# -------------------------------------------------------------
from __future__ import annotations
from typing import Optional
import base64

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
import pytz

# ---------- BASIS ----------
st.set_page_config(page_title="ABBA noneq — v7.7", layout="wide")
LIZARD_BASE = "https://flevoland.lizard.net/api/v4"
RASTER_UUID_PRECIP = "730d6675-35dd-4a35-aa9b-bfb8155f9ca7"  # 5-min neerslagraster
EU_TZ = pytz.timezone("Europe/Amsterdam")
SESSION = requests.Session()

# ---------- Sidebar (deel 1): auth + locatie ----------
with st.sidebar:
    st.header("Authenticatie & locatie")

    # ✅ Defensief: None -> "" zodat .strip() nooit crasht
    prev_token = st.session_state.get("__lizard_token__", "")
    token_input = st.text_input("API key (verplicht)", value=prev_token or "", type="password")
    token = (token_input or "").strip()
    st.session_state["__lizard_token__"] = token or None

    prev_prefix = st.session_state.get("__abba_prefix__", "212_1")
    prefix_input = st.text_input("Locatieprefix", value=prev_prefix or "212_1")
    CODE_PREFIX = (prefix_input or "212_1").strip()
    st.session_state["__abba_prefix__"] = CODE_PREFIX

    if not token:
        st.error("Voer een geldige API key in om verder te gaan.")
        st.stop()

st.title(f"ABBA-berekening voor gemaal {CODE_PREFIX}")
st.caption("voer de ABBA-berekening per gemaal uit voor je eigen gekozen periode. Vergelijk de data en exporteer eventueel.")

# ---------- Helpers ----------
def _headers() -> dict:
    up = f"__key__:{(st.session_state.get('__lizard_token__') or '').strip()}"
    return {"Accept": "application/json", "Authorization": f"Basic {base64.b64encode(up.encode()).decode()}"}

def _get(url: str, params: Optional[dict] = None, timeout: int = 30) -> dict:
    r = SESSION.get(url, params=params, headers=_headers(), timeout=timeout)
    if r.status_code == 401:
        st.error("401 Unauthorized — controleer je API key.")
        st.stop()
    r.raise_for_status()
    return r.json()

def _to_local(ts):
    s = pd.to_datetime(ts, utc=True)
    return s.dt.tz_convert(EU_TZ) if isinstance(s, pd.Series) else s.tz_convert(EU_TZ)

def _clip(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].clip(lower=lo, upper=hi)

# ---------- API helpers ----------
@st.cache_data(ttl=300)
def list_timeseries_for_prefix(prefix: str) -> pd.DataFrame:
    url = f"{LIZARD_BASE}/timeseries/"
    params = {
        "location__code__startswith": prefix,
        "page_size": 1000,
        "expand": "observation_type,location",
    }
    rows = []
    while True:
        data = _get(url, params)
        for ts in data.get("results", []):
            obs = ts.get("observation_type") or {}
            param_name = ""
            unit_name = ""
            if isinstance(obs, dict):
                p = obs.get("parameter"); u = obs.get("unit")
                if isinstance(p, dict):
                    param_name = (p.get("name") or p.get("code") or "").strip()
                elif isinstance(p, str):
                    param_name = p.strip()
                if isinstance(u, dict):
                    unit_name = (u.get("name") or u.get("symbol") or "").strip()
                elif isinstance(u, str):
                    unit_name = u.strip()
            label = (ts.get("label") or ts.get("name") or ts.get("code") or "").strip()
            code  = (ts.get("code") or "").strip()
            loc   = ts.get("location") or {}
            loc_code = (loc.get("code") or "").strip() if isinstance(loc, dict) else ""
            loc_uuid = (loc.get("uuid") or "").strip() if isinstance(loc, dict) else ""
            base = param_name or label or code or ts.get("uuid", "")
            nice = f"{loc_code} · {base}" if loc_code else base
            if code:
                nice = f"{nice} ({code})"
            rows.append({
                "uuid": ts.get("uuid"),
                "param": (param_name or "").lower(),
                "unit": unit_name,
                "label": label,
                "code": code,
                "location": loc_code,
                "location_uuid": loc_uuid,
                "nice": nice,
            })
        next_url = data.get("next")
        if not next_url:
            break
        url, params = next_url, None
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def get_lt5_metadata(prefix: str) -> dict:
    exact = _get(f"{LIZARD_BASE}/locations/", {"code": f"{prefix}#LT5", "page_size": 1})
    results = exact.get("results", [])
    if not results:
        starts = _get(f"{LIZARD_BASE}/locations/", {"code__startswith": f"{prefix}%23LT5", "page_size": 1})
        results = starts.get("results", [])
        if not results:
            return {}
    detail = _get(f"{LIZARD_BASE}/locations/{results[0]['uuid']}/")
    return detail.get("extra_metadata") or {}

@st.cache_data(ttl=300)
def get_location_detail(loc_uuid: str) -> dict:
    if not loc_uuid:
        return {}
    return _get(f"{LIZARD_BASE}/locations/{loc_uuid}/")

@st.cache_data(ttl=300)
def fetch_events(ts_uuid: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    if not ts_uuid:
        return pd.DataFrame(columns=["time","value"])
    url = f"{LIZARD_BASE}/timeseries/{ts_uuid}/events/"
    params = {"time__gte": start_iso, "time__lte": end_iso, "page_size": 2000}
    rows = []
    while True:
        data = _get(url, params)
        for e in data.get("results", []):
            rows.append((pd.to_datetime(e["time"], utc=True), e.get("value")))
        next_url = data.get("next")
        if not next_url:
            break
        url, params = next_url, None
    df = pd.DataFrame(rows, columns=["time","value"]) if rows else pd.DataFrame(columns=["time","value"])
    if not df.empty:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("time")
    return df

@st.cache_data(ttl=300)
def fetch_precip_raster(lat: float, lon: float, start_iso: str, end_iso: str, interval: str = "5m") -> pd.DataFrame:
    url = f"{LIZARD_BASE}/rasters/{RASTER_UUID_PRECIP}/point/"
    params = {"geom": f"POINT({lon} {lat})", "srs": "EPSG:4326", "start": start_iso, "end": end_iso, "interval": interval}
    data = _get(url, params)
    if isinstance(data, dict):
        if "values" in data and isinstance(data["values"], list):
            df = pd.DataFrame(data["values"], columns=["time","neerslag"])
        elif "results" in data and isinstance(data["results"], list):
            df = pd.DataFrame([(row.get("time"), row.get("value")) for row in data["results"]], columns=["time","neerslag"])
        else:
            df = pd.DataFrame(columns=["time","neerslag"])
    else:
        df = pd.DataFrame(columns=["time","neerslag"])
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["neerslag"] = pd.to_numeric(df["neerslag"], errors="coerce")
        df = df.dropna().sort_values("time")
    return df

# ---------- ABBA noneq ----------
def waterhoogte_to_debiet_noneq(df_h: pd.DataFrame, A_peil: float, B_peil: float, opp: float, s_u: float = 1.0) -> pd.DataFrame:
    """Netto debiet (m³/uur) uit waterhoogte. Positief = instroom."""
    if df_h is None or df_h.empty:
        return pd.DataFrame(columns=["time","debiet","debiet_in","debiet_uit"])
    d = df_h.sort_values("time").copy()
    d["clip"] = d["value"].clip(lower=B_peil, upper=A_peil)
    d["dt"] = pd.to_datetime(d["time"], utc=True).diff().dt.total_seconds()
    d = d[d["dt"] > 0].copy()
    d["gradient"] = d["clip"].diff() / d["dt"]
    d["debiet"] = d["gradient"] * float(s_u) * float(opp) * 3600.0
    d = d.dropna(subset=["debiet"])
    d["debiet_in"]  = np.where(d["debiet"] > 0, d["debiet"], np.nan)   # v7.2-gedrag, géén sign-flip
    d["debiet_uit"] = np.where(d["debiet"] < 0, d["debiet"], np.nan)
    return d[["time","debiet","debiet_in","debiet_uit"]]

def bedrijf_from_status(df_status: pd.DataFrame, label: str, *, use_bitmask: bool, bit_index: int, threshold: float) -> pd.DataFrame:
    """Vectorized 0/1 uit status: bitmask (NumPy right_shift) of threshold."""
    if df_status is None or df_status.empty:
        return pd.DataFrame(columns=["time", f"bedrijf_{label}"])
    d = df_status.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    s = pd.to_numeric(d["value"], errors="coerce")

    if use_bitmask:
        iv = s.fillna(0).astype(np.int64)
        shifted = np.right_shift(iv.to_numpy(), int(bit_index))
        onoff = pd.Series((shifted & 1), index=d.index, dtype="int64").astype(float)
    else:
        onoff = (s > float(threshold)).astype(float)

    d[f"bedrijf_{label}"] = onoff.round(0)  # force 0/1
    d = d.dropna(subset=[f"bedrijf_{label}"]).drop_duplicates("time", keep="last")
    return d[["time", f"bedrijf_{label}"]].sort_values("time")

def abba_berekenen_noneq(df_h, df_p1, df_p2, A_peil, B_peil, opp,
                         start_utc, end_utc, s_u,
                         *, use_bitmask_p1, bit_index_p1, thr_p1,
                            use_bitmask_p2, bit_index_p2, thr_p2) -> pd.DataFrame:
    d = waterhoogte_to_debiet_noneq(df_h, A_peil, B_peil, opp, s_u=s_u)
    if d.empty:
        return d
    d = d[(d["time"] >= start_utc) & (d["time"] <= end_utc)].copy()
    if d.empty:
        return d

    p1 = bedrijf_from_status(df_p1, "P1", use_bitmask=use_bitmask_p1, bit_index=bit_index_p1, threshold=thr_p1)
    p2 = bedrijf_from_status(df_p2, "P2", use_bitmask=use_bitmask_p2, bit_index=bit_index_p2, threshold=thr_p2) if df_p2 is not None else pd.DataFrame()

    d = d.sort_values("time")
    if not p1.empty:
        d = pd.merge_asof(d, p1.sort_values("time"), on="time", direction="backward")
    else:
        d["bedrijf_P1"] = 0.0
    if not p2.empty:
        d = pd.merge_asof(d, p2.sort_values("time"), on="time", direction="backward")
    else:
        d["bedrijf_P2"] = 0.0

    d["bedrijf_P1"] = d["bedrijf_P1"].fillna(0.0).round(0)
    d["bedrijf_P2"] = d["bedrijf_P2"].fillna(0.0).round(0)

    # v7.2-gedrag: pompdebiet = status × netto debiet, maar alleen bij uitstroom (debiet < 0)
    d["debiet_pomp_P1"] = np.where(d["debiet"] < 0, d["bedrijf_P1"] * d["debiet"], 0.0)
    d["debiet_pomp_P2"] = np.where(d["debiet"] < 0, d["bedrijf_P2"] * d["debiet"], 0.0)

    return d

# ---------- Sidebar (deel 2): periode & s_u & bits ----------
with st.sidebar:
    st.header("Periode & parameters")
    today_local = pd.Timestamp.now(tz=EU_TZ).floor("D")
    default_start = today_local - pd.Timedelta(days=3)
    start_date = st.date_input("Startdatum", value=default_start.date())
    end_date   = st.date_input("Einddatum",   value=today_local.date())
    start_utc = pd.Timestamp(start_date, tz=EU_TZ).tz_convert("UTC").floor("D")
    end_utc   = (pd.Timestamp(end_date, tz=EU_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_convert("UTC")
    s_u = st.number_input("Schaalfactor s_u", value=1.0, min_value=0.0, step=0.1)

    # st.subheader("Status-interpretatie per pomp")
    # # ✅ default voor beide pompen = bit 9 (jouw wens)
    # use_bitmask_p1 = st.checkbox("P1 gebruikt bitmask", value=True)
    # bit_index_p1   = st.number_input("Bit-index P1 (0=LSB)", value=9, min_value=0, step=1, disabled=not use_bitmask_p1)
    # thr_p1         = st.number_input("Drempel P1 (> betekent aan)", value=0.0, step=0.001, disabled=use_bitmask_p1)

    # use_bitmask_p2 = st.checkbox("P2 gebruikt bitmask", value=True)
    # bit_index_p2   = st.number_input("Bit-index P2 (0=LSB)", value=9, min_value=0, step=1, disabled=not use_bitmask_p2)
    # thr_p2         = st.number_input("Drempel P2 (> betekent aan)", value=0.0, step=0.001, disabled=use_bitmask_p2)

# ---------- Tijdserieslijst + selectie ----------
with st.spinner("Tijdseries ophalen…"):
    ts_all = list_timeseries_for_prefix(CODE_PREFIX)
if ts_all.empty:
    st.warning(f"Geen tijdseries gevonden voor {CODE_PREFIX}.")
    st.stop()

def _match_any(s, needles): return any(n in (s or "").lower() for n in needles)

is_water = ts_all["param"].apply(lambda p: _match_any(p, ["waterhoogte","waterstand","nap"])) \
           | ts_all["label"].apply(lambda p: _match_any(p, ["waterhoogte","waterstand","nap"]))
water_df = ts_all[is_water].copy()
water_df = pd.concat([water_df[water_df["location"].str.contains("LT5", case=False, na=False)],
                      water_df[~water_df["location"].str.contains("LT5", case=False, na=False)]])

is_status = ts_all["param"].apply(lambda p: _match_any(p, ["pompstatus","status"])) \
            | ts_all["label"].apply(lambda p: _match_any(p, ["pompstatus","status"]))
p1_df = ts_all[is_status].copy()
p1_df = pd.concat([p1_df[p1_df["location"].str.contains("P1", case=False, na=False)],
                   p1_df[~p1_df["location"].str.contains("P1", case=False, na=False)]])
p2_df = ts_all[is_status].copy()
p2_df = pd.concat([p2_df[p2_df["location"].str.contains("P2", case=False, na=False)],
                   p2_df[~p2_df["location"].str.contains("P2", case=False, na=False)]])

if water_df.empty: water_df = ts_all.copy()
if p1_df.empty:    p1_df = ts_all.copy()
if p2_df.empty:    p2_df = ts_all.copy()

# ---------- Tijdseries selecteren (met slimme defaults) ----------
# ---------- Tijdseries selecteren (met slimme defaults en Python-int index) ----------

def _first_pos(mask: pd.Series) -> int | None:
    """Geef de 0-based positie binnen de gefilterde DataFrame-orde terug als pure int."""
    if mask is None or mask.empty or not mask.any():
        return None
    # vind het eerste True-label en reken om naar positie t.o.v. de DataFrame-volgorde
    first_label = mask[mask].index[0]
    # positioneel nummer bepalen: gebruik get_indexer voor een pure int
    pos_arr = mask.index.get_indexer([first_label])
    return int(pos_arr[0]) if len(pos_arr) and pos_arr[0] >= 0 else None

# Waterhoogte (LT5)
water_opts = ["— kies —"] + water_df["nice"].tolist()
default_water_idx = 1 if len(water_opts) > 1 else 0          # eerste echte optie
# (optioneel nog strenger LT5 zoeken)
if not water_df.empty:
    lt5_mask = water_df["location"].str.contains("LT5", case=False, na=False)
    pos = _first_pos(lt5_mask)
    if pos is not None:
        default_water_idx = 1 + pos
default_water_idx = int(max(0, min(default_water_idx, len(water_opts) - 1)))  # pure int + bounds

sel_water = st.selectbox("Waterhoogte (LT5)", water_opts, index=default_water_idx)

# Pompstatus P1
p1_opts = ["— kies —"] + p1_df["nice"].tolist()
if p1_df.empty:
    default_p1_idx = 0
else:
    p1_mask = p1_df["location"].str.contains("P1", case=False, na=False)
    pos = _first_pos(p1_mask)
    default_p1_idx = 1 + pos if pos is not None else 1
default_p1_idx = int(max(0, min(default_p1_idx, len(p1_opts) - 1)))

sel_p1 = st.selectbox("Pompstatus P1", p1_opts, index=default_p1_idx)

# Pompstatus P2 (optioneel)
p2_opts = ["— geen —"] + p2_df["nice"].tolist()
default_p2_idx = 0  # "— geen —"
if not p2_df.empty:
    p2_mask = p2_df["location"].str.contains("P2", case=False, na=False)
    pos = _first_pos(p2_mask)
    if pos is not None:
        default_p2_idx = 1 + pos
default_p2_idx = int(max(0, min(default_p2_idx, len(p2_opts) - 1)))

sel_p2 = st.selectbox("Pompstatus P2 (optioneel)", p2_opts, index=default_p2_idx)

if sel_water.startswith("—") or sel_p1.startswith("—"):
    st.info("Kies minstens Waterhoogte en Pompstatus P1.")
    st.stop()

row_w = water_df.loc[water_df["nice"]==sel_water].iloc[0]
uuid_water = row_w["uuid"]; loc_uuid_water = row_w.get("location_uuid", "")

uuid_p1 = p1_df.loc[p1_df["nice"]==sel_p1, "uuid"].iloc[0]
uuid_p2 = p2_df.loc[p2_df["nice"]==sel_p2, "uuid"].iloc[0] if sel_p2 != "— geen —" and (p2_df["nice"]==sel_p2).any() else None

with st.spinner("Inputdata ophalen…"):
    df_h  = fetch_events(uuid_water, start_utc.isoformat(), end_utc.isoformat())
    df_p1 = fetch_events(uuid_p1,    start_utc.isoformat(), end_utc.isoformat())
    df_p2 = fetch_events(uuid_p2,    start_utc.isoformat(), end_utc.isoformat()) if uuid_p2 else pd.DataFrame()

# ---------- Inputdata tonen ----------
with st.expander("Inputdata (eerste 20 rijen)"):
    colA, colB, colC = st.columns(3)
    with colA: st.write("Waterhoogte (LT5)"); st.dataframe(df_h.head(20))
    with colB: st.write("Pompstatus P1");    st.dataframe(df_p1.head(20))
    with colC: st.write("Pompstatus P2");    st.dataframe(df_p2.head(20) if not df_p2.empty else pd.DataFrame(columns=["time","value"]))

# ---------- LT5 metadata (optioneel A/B/opp) ----------
extra = get_lt5_metadata(CODE_PREFIX)
def _as_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

st.subheader("ABBA-parameters (LT5 metadata)")
col1, col2, col3 = st.columns(3)
with col1: A_peil = st.number_input("A-peil (max)", value=_as_float(extra.get("meetniveau_abba_a"), 0.0), step=0.01)
with col2: B_peil = st.number_input("B-peil (min)", value=_as_float(extra.get("meetniveau_abba_b"), 0.0), step=0.01)
with col3: opp    = st.number_input("Oppervlakte (m²)", value=_as_float(extra.get("oppervlakte"), 0.0), min_value=0.0, step=0.001)

use_bitmask_p1 = True
bit_index_p1   = 9
thr_p1         = 0.0

use_bitmask_p2 = True
bit_index_p2   = 9
thr_p2         = 0.0
# ---------- ABBA-berekening ----------
with st.spinner("ABBA-debieten (noneq) berekenen…"):
    df_out = abba_berekenen_noneq(
        df_h=df_h, df_p1=df_p1, df_p2=df_p2 if not df_p2.empty else None,
        A_peil=A_peil, B_peil=B_peil, opp=opp,
        start_utc=start_utc, end_utc=end_utc, s_u=float(s_u),
        use_bitmask_p1=use_bitmask_p1, bit_index_p1=int(bit_index_p1), thr_p1=float(thr_p1),
        use_bitmask_p2=use_bitmask_p2, bit_index_p2=int(bit_index_p2), thr_p2=float(thr_p2),
    )

if df_out.empty:
    st.warning("Geen resultaten (controleer inputreeksen, periode of metadata).")
    st.stop()

# ---------- Neerslag (precip_cum) ophalen/opbouwen ----------
precip_cum = None
try:
    loc_detail = get_location_detail(loc_uuid_water) if loc_uuid_water else {}
    geom = (loc_detail.get("geometry") or {}).get("coordinates") if isinstance(loc_detail.get("geometry"), dict) else None
    if isinstance(geom, (list, tuple)) and len(geom) >= 2:
        lon, lat = float(geom[0]), float(geom[1])
        pr = fetch_precip_raster(lat, lon, start_utc.isoformat(), end_utc.isoformat(), interval="5m")
        if not pr.empty:
            pr["time_local"] = _to_local(pr["time"])
            pr["dag"] = pr["time_local"].dt.date
            pr["neerslag_cum_dag"] = pr.groupby("dag")["neerslag"].cumsum()
            precip_cum = pr.drop(columns=["dag"])
except Exception:
    precip_cum = None  # stil vallen: lege neerslaggrafiek

# ---------- plot_df samenstellen ----------
plot_df = df_out.copy()
plot_df["time_local"] = _to_local(plot_df["time"])

# pompreeksen positief tekenen (alleen visueel flip van het teken)
if "debiet_pomp_P1" in plot_df.columns:
    plot_df["debiet_pomp_P1_pos"] = -1.0 * plot_df["debiet_pomp_P1"]
if "debiet_pomp_P2" in plot_df.columns:
    plot_df["debiet_pomp_P2_pos"] = -1.0 * plot_df["debiet_pomp_P2"]

# ---------- Sidebar (deel 3): drempels (clip alléén de debiet-lijn) ----------
with st.sidebar:
    st.header("Grafiek-drempels")
    apply_clip = st.checkbox("Drempel-filter voor debietlijn toepassen", value=False)

    base_series = plot_df["debiet_in"].dropna() if "debiet_in" in plot_df else pd.Series([], dtype=float)
    if base_series.empty and "debiet" in plot_df.columns:
        base_series = plot_df["debiet"].dropna()
    if base_series.empty:
        base_series = pd.Series([-100, 100], dtype=float)

    p_lo = float(base_series.quantile(0.05)); p_hi = float(base_series.quantile(0.95))
    d_lo = st.number_input("Debietlijn min (m³/uur)", value=round(p_lo, 2))
    d_hi = st.number_input("Debietlijn max (m³/uur)", value=round(p_hi, 2))

    if apply_clip:
        if "debiet_in" in plot_df.columns:
            plot_df["debiet_in"] = plot_df["debiet_in"].clip(lower=d_lo, upper=d_hi)
        if "debiet" in plot_df.columns:
            plot_df["debiet"] = plot_df["debiet"].clip(lower=d_lo, upper=d_hi)

    # ⚠️ niet clippen: 'debiet_pomp_P1_pos' / 'debiet_pomp_P2_pos'

# ---------- Plotfuncties ----------
def plot_bedrijf(df: pd.DataFrame, height: int = 240) -> alt.Chart:
    want = {"bedrijf_p1", "bedrijf_p2"}
    cols = [c for c in df.columns if c.lower() in want]
    if not cols:
        return alt.Chart(pd.DataFrame({"time_local": [], "waarde": []})).mark_line().properties(height=height)
    b = df[["time"] + cols].copy()
    for c in cols:
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0).round(0)

    SCALE = 1000.0  # alleen visualisatie
    for c in cols:
        b[c] = b[c] * SCALE

    b["time_local"] = _to_local(b["time"])
    long = b.melt("time_local", var_name="serie", value_name="waarde")
    return (
        alt.Chart(long)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("time_local:T", title="Tijd"),
            y=alt.Y("waarde:Q", title="Bedrijf (0/1) ×1000", scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color("serie:N", legend=alt.Legend(orient="right", title="Bedrijf")),
            tooltip=["time_local:T", "serie:N", alt.Tooltip("waarde:Q", title="waarde (×1000)")],
        )
        .properties(height=height)
    )

def plot_debiet(df: pd.DataFrame, height: int = 240) -> alt.Chart:
    cols = ["time_local", "debiet_in", "debiet", "debiet_pomp_P1_pos", "debiet_pomp_P2_pos"]
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()

    parts = []
    line_src = d[["time_local", "debiet_in"]].dropna() if "debiet_in" in d.columns else pd.DataFrame()
    line_label = "debiet_in"
    if line_src.empty and "debiet" in d.columns:
        line_src = d[["time_local", "debiet"]].dropna().rename(columns={"debiet": "debiet_in"})
        line_label = "debiet"
    if not line_src.empty:
        parts.append(line_src.rename(columns={"debiet_in": "waarde"}).assign(serie=line_label))

    if "debiet_pomp_P1_pos" in d.columns:
        parts.append(d[["time_local", "debiet_pomp_P1_pos"]].dropna().rename(columns={"debiet_pomp_P1_pos":"waarde"}).assign(serie="pomp1"))
    if "debiet_pomp_P2_pos" in d.columns:
        parts.append(d[["time_local", "debiet_pomp_P2_pos"]].dropna().rename(columns={"debiet_pomp_P2_pos":"waarde"}).assign(serie="pomp2"))

    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"time_local": [], "waarde": [], "serie": []})

    # Alleen geldige reeksen in de plot/legenda
    allowed = {line_label, "pomp1", "pomp2"}
    long = long[long["serie"].isin(allowed)]

    line = (
        alt.Chart(long[long["serie"] == line_label])
        .mark_line()
        .encode(
            x=alt.X("time_local:T", title="Tijd"),
            y=alt.Y("waarde:Q", title="Debiet (m³/uur)"),
            color=alt.Color("serie:N", legend=alt.Legend(orient="right", title="Reeks")),
            tooltip=["time_local:T", "serie:N", "waarde:Q"],
        )
        .properties(height=height)
    )
    pts = (
        alt.Chart(long[long["serie"].isin(["pomp1","pomp2"])])
        .mark_point()
        .encode(
            x="time_local:T",
            y="waarde:Q",
            color=alt.Color("serie:N", legend=alt.Legend(orient="right", title="Reeks")),
            tooltip=["time_local:T", "serie:N", "waarde:Q"],
        )
        .properties(height=height)
    )
    return line + pts

def plot_neerslag(precip_cum: pd.DataFrame | None, height: int = 240) -> alt.Chart:
    if precip_cum is None or precip_cum.empty or "time_local" not in precip_cum.columns:
        return alt.Chart(pd.DataFrame({"time_local": [], "neerslag_cum_dag": []})).mark_line().properties(height=height)
    d = precip_cum.copy()
    if "neerslag_cum_dag" not in d.columns and "neerslag" in d.columns:
        d["dag"] = d["time_local"].dt.date
        d["neerslag_cum_dag"] = d.groupby("dag")["neerslag"].cumsum()
        d = d.drop(columns=["dag"])
    if "neerslag_cum_dag" not in d.columns:
        return alt.Chart(pd.DataFrame({"time_local": [], "neerslag_cum_dag": []})).mark_line().properties(height=height)
    return (
        alt.Chart(d)
        .mark_line()
        .encode(
            x=alt.X("time_local:T", title="Tijd"),
            y=alt.Y("neerslag_cum_dag:Q", title="Neerslag cumulatief [mm/dag]"),
            tooltip=["time_local:T",
                     alt.Tooltip("neerslag_cum_dag:Q", title="cum dag"),
                     alt.Tooltip("neerslag:Q", title="5-min") if "neerslag" in d.columns else alt.value(None)],
        )
        .properties(height=height)
    )

# ---------- RENDER ----------
zoom = alt.selection_interval(bind="scales", encodings=["x"])
c_bedrijf = plot_bedrijf(plot_df, height=260).add_params(zoom)
c_debiet  = plot_debiet(plot_df,  height=260).add_params(zoom)
c_rain    = plot_neerslag(precip_cum, height=260).add_params(zoom)
combo = alt.vconcat(c_bedrijf, c_debiet, c_rain).resolve_scale(x="shared")
st.altair_chart(combo, use_container_width=True)
