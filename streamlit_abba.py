# streamlit_abba_noneq_v_7_9.py
# -------------------------------------------------------------
# ABBA noneq — Cloud-ready met extras:
# - Pumpstations (Almere + Urk) ophalen en kiezen (code of naam)
# - Prefix automatisch uit keuze
# - Vast: bitmask bit 9 voor P1 en P2
# - s_u verwijderd (vast 1.0)
# - debiet_in uit waterhoogte (positief = instroom)
# - pompdebiet = status × debiet (alleen bij uitstroom, dus debiet<0), positief geplot
# - status ×1000 visueel
# - KPI's: draaiuren, schakelingen, debiet/draaiuren
# - Export naar Excel (inputs, outputs, KPIs)
# - Clipping alleen voor debiet-lijn
# -------------------------------------------------------------
from __future__ import annotations
from typing import Optional, Tuple
import io
import base64

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

import streamlit as st
import altair as alt
import pytz

# ---------- BASIS ----------
st.set_page_config(page_title="ABBA noneq — v7.9", layout="wide")
EU_TZ = pytz.timezone("Europe/Amsterdam")
LIZARD_BASE = "https://flevoland.lizard.net/api/v4"
RASTER_UUID_PRECIP = "730d6675-35dd-4a35-aa9b-bfb8155f9ca7"  # 5-min neerslag

# HTTP session met retries/backoff
SESSION = requests.Session()
_retry = Retry(
    total=3, connect=3, read=3, status=3,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

# ---------- Helper utils ----------
def _headers() -> dict:
    up = f"__key__:{(st.session_state.get('__lizard_token__') or '').strip()}"
    return {"Accept": "application/json", "Authorization": f"Basic {base64.b64encode(up.encode()).decode()}"}

def _get(url: str, params: Optional[dict] = None, timeout: Tuple[float,float] = (5.0, 25.0)) -> dict:
    try:
        r = SESSION.get(url, params=params, headers=_headers(), timeout=timeout)
        if r.status_code == 401:
            st.error("401 Unauthorized — controleer je API key.")
            st.stop()
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectTimeout:
        st.error("Timeout bij verbinden met Lizard (connect).")
        st.stop()
    except requests.exceptions.ReadTimeout:
        st.error("Timeout bij lezen van Lizard (read).")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Netwerkfout naar Lizard: {e}")
        st.stop()

def _to_local(ts):
    s = pd.to_datetime(ts, utc=True)
    return s.dt.tz_convert(EU_TZ) if isinstance(s, pd.Series) else s.tz_convert(EU_TZ)

def _clip(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].clip(lower=lo, upper=hi)

# ---------- API helpers ----------
@st.cache_data(ttl=300)
def list_pumpstations() -> pd.DataFrame:
    """Alle gemalen (Almere + Urk) met code en name."""
    url = f"{LIZARD_BASE}/pumpstations/"
    params = {"organisation__name__in": "Almere,Urk", "page_size": 100}
    rows = []
    while True:
        data = _get(url, params)
        for p in data.get("results", []):
            rows.append({
                "uuid": p.get("uuid"),
                "code": (p.get("code") or "").strip(),
                "name": (p.get("name") or "").strip(),
            })
        next_url = data.get("next")
        if not next_url:
            break
        url, params = next_url, None
    df = pd.DataFrame(rows)
    if not df.empty:
        df["label_code"] = df["code"] + " — " + df["name"]
        df["label_name"] = df["name"] + " — " + df["code"]
    return df

@st.cache_data(ttl=300)
def list_timeseries_for_prefix(prefix: str) -> pd.DataFrame:
    url = f"{LIZARD_BASE}/timeseries/"
    params = {
        "location__code__startswith": prefix,
        "page_size": 500,
        "expand": "observation_type,location",
    }
    rows = []
    while True:
        data = _get(url, params)
        for ts in data.get("results", []):
            obs = ts.get("observation_type") or {}
            param_name, unit_name = "", ""
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
def waterhoogte_to_debiet_noneq(df_h: pd.DataFrame, A_peil: float, B_peil: float, opp: float) -> pd.DataFrame:
    """Netto debiet (m³/uur) uit waterhoogte. Positief = instroom."""
    if df_h is None or df_h.empty:
        return pd.DataFrame(columns=["time","debiet","debiet_in","debiet_uit"])
    d = df_h.sort_values("time").copy()
    d["clip"] = d["value"].clip(lower=B_peil, upper=A_peil)
    d["dt"] = pd.to_datetime(d["time"], utc=True).diff().dt.total_seconds()
    d = d[d["dt"] > 0].copy()
    d["gradient"] = d["clip"].diff() / d["dt"]
    d["debiet"] = d["gradient"] * 1.0 * float(opp) * 3600.0  # s_u = 1.0
    d = d.dropna(subset=["debiet"])
    d["debiet_in"]  = np.where(d["debiet"] > 0, d["debiet"], np.nan)
    d["debiet_uit"] = np.where(d["debiet"] < 0, d["debiet"], np.nan)
    return d[["time","debiet","debiet_in","debiet_uit"]]

def bedrijf_from_status(df_status: pd.DataFrame, label: str, *, bit_index: int) -> pd.DataFrame:
    """Vectorized 0/1 uit status: bitmask (NumPy right_shift) op vast bit_index."""
    if df_status is None or df_status.empty:
        return pd.DataFrame(columns=["time", f"bedrijf_{label}"])
    d = df_status.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    s = pd.to_numeric(d["value"], errors="coerce").fillna(0).astype(np.int64)
    shifted = np.right_shift(s.to_numpy(), int(bit_index))
    onoff = pd.Series((shifted & 1), index=d.index, dtype="int64").astype(float)
    d[f"bedrijf_{label}"] = onoff.round(0)
    d = d.dropna(subset=[f"bedrijf_{label}"]).drop_duplicates("time", keep="last")
    return d[["time", f"bedrijf_{label}"]].sort_values("time")

def abba_berekenen_noneq(df_h, df_p1, df_p2, A_peil, B_peil, opp,
                         start_utc, end_utc, *,
                         bit_index_p1: int, bit_index_p2: int) -> pd.DataFrame:
    d = waterhoogte_to_debiet_noneq(df_h, A_peil, B_peil, opp)
    if d.empty:
        return d
    d = d[(d["time"] >= start_utc) & (d["time"] <= end_utc)].copy()
    if d.empty:
        return d

    p1 = bedrijf_from_status(df_p1, "P1", bit_index=bit_index_p1)
    p2 = bedrijf_from_status(df_p2, "P2", bit_index=bit_index_p2) if df_p2 is not None and not df_p2.empty else pd.DataFrame()

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

# ---------- Sidebar: auth + selectie + periode ----------
with st.sidebar:
    st.header("Authenticatie & locatie")

    # ✅ Defensief: None -> "" zodat .strip() nooit crasht
    prev_token = st.session_state.get("__lizard_token__", "")
    token_input = st.text_input("API key (verplicht)", value=prev_token or "", type="password")
    token = (token_input or "").strip()
    st.session_state["__lizard_token__"] = token or None

    # prev_prefix = st.session_state.get("__abba_prefix__", "212_1")
    # prefix_input = st.text_input("Locatieprefix", value=prev_prefix or "212_1")
    # CODE_PREFIX = (prefix_input or "212_1").strip()
    # st.session_state["__abba_prefix__"] = CODE_PREFIX

    if not token:
        st.error("Voer een geldige API key in om verder te gaan.")
        st.stop()

    st.header("Gemaal kiezen")
    pumps = list_pumpstations()
    by = st.radio("Selecteer op", ["code", "naam"], index=0, horizontal=True)
    if pumps.empty:
        st.warning("Geen gemalen gevonden (Almere/Urk). Voer desnoods handmatig een prefix in.")
        prefix_input = st.text_input("Locatieprefix (fallback)", value=str(st.session_state.get("__abba_prefix__", "115_8") or "115_8"))
        CODE_PREFIX = (prefix_input or "115_8").strip()
    else:
        opts = pumps["label_code"].tolist() if by == "code" else pumps["label_name"].tolist()
        default_idx = int(0 if not opts else 0)
        sel = st.selectbox("Gemaal", opts, index=default_idx)
        row = pumps.iloc[opts.index(sel)]
        # Neem de pumpstation code als prefix (zoals 115_8 etc.)
        CODE_PREFIX = row["code"]
    st.session_state["__abba_prefix__"] = CODE_PREFIX

    st.header("Periode")
    today_local = pd.Timestamp.now(tz=EU_TZ).floor("D")
    default_start = today_local - pd.Timedelta(days=3)
    start_date = st.date_input("Startdatum", value=default_start.date())
    end_date   = st.date_input("Einddatum",   value=today_local.date())
    start_utc = pd.Timestamp(start_date, tz=EU_TZ).tz_convert("UTC").floor("D")
    end_utc   = (pd.Timestamp(end_date, tz=EU_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_convert("UTC")

# st.write(f"**Prefix:** `{CODE_PREFIX}`")

# ---------- Header met logo's ----------
left, mid, right = st.columns([1,4,1])
with left:
    # Placeholder-logo's; vervang door echte URLs
    st.image("https://www.almere.nl/_assets/f20fb53eac3716ad4f374a5272077327/Images/logo-almere-turquoise.svg", caption="Almere", use_container_width=True)
with mid:
    st.title(f"ABBA-berekening voor gemaal {CODE_PREFIX}")
    st.caption("voer de ABBA-berekening per gemaal uit voor je eigen gekozen periode. Vergelijk de data en exporteer eventueel.")
with right:
    st.image("https://nelen-schuurmans.nl/uploads/Lizard-header-case-1.jpg", caption="Lizard", use_container_width=True)

# ---------- Tijdserieslijst + slimme defaults ----------
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

def _first_pos(mask: pd.Series) -> int | None:
    if mask is None or mask.empty or not mask.any():
        return None
    first_label = mask[mask].index[0]
    pos_arr = mask.index.get_indexer([first_label])
    return int(pos_arr[0]) if len(pos_arr) and pos_arr[0] >= 0 else None

# Waterhoogte
water_opts = ["— kies —"] + water_df["nice"].tolist()
default_water_idx = 1 if len(water_opts) > 1 else 0
if not water_df.empty:
    lt5_mask = water_df["location"].str.contains("LT5", case=False, na=False)
    pos = _first_pos(lt5_mask)
    if pos is not None:
        default_water_idx = 1 + pos
default_water_idx = int(max(0, min(default_water_idx, len(water_opts) - 1)))
sel_water = st.selectbox("Waterhoogte (LT5)", water_opts, index=default_water_idx)

# P1
p1_opts = ["— kies —"] + p1_df["nice"].tolist()
if p1_df.empty:
    default_p1_idx = 0
else:
    p1_mask = p1_df["location"].str.contains("P1", case=False, na=False)
    pos = _first_pos(p1_mask)
    default_p1_idx = 1 + pos if pos is not None else 1
default_p1_idx = int(max(0, min(default_p1_idx, len(p1_opts) - 1)))
sel_p1 = st.selectbox("Pompstatus P1", p1_opts, index=default_p1_idx)

# P2 (optioneel)
p2_opts = ["— geen —"] + p2_df["nice"].tolist()
default_p2_idx = 0
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

row_w = water_df.loc[water_df["nice"] == sel_water].iloc[0]
uuid_water = row_w["uuid"]; loc_uuid_water = row_w.get("location_uuid", "")

uuid_p1 = p1_df.loc[p1_df["nice"] == sel_p1, "uuid"].iloc[0]
uuid_p2 = p2_df.loc[p2_df["nice"] == sel_p2, "uuid"].iloc[0] if sel_p2 != "— geen —" and (p2_df["nice"]==sel_p2).any() else None

# ---------- Inputdata ophalen ----------
with st.spinner("Inputdata ophalen…"):
    df_h  = fetch_events(uuid_water, start_utc.isoformat(), end_utc.isoformat())
    df_p1 = fetch_events(uuid_p1,    start_utc.isoformat(), end_utc.isoformat())
    df_p2 = fetch_events(uuid_p2,    start_utc.isoformat(), end_utc.isoformat()) if uuid_p2 else pd.DataFrame()

# ---------- Metadata A/B/opp ----------
extra = get_lt5_metadata(CODE_PREFIX)

def _as_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

st.subheader("ABBA-parameters (LT5)")
col1, col2, col3 = st.columns(3)

with col1:
    A_peil = st.number_input(
        "A-peil (max) [m NAP]",
        value=_as_float(extra.get("meetniveau_abba_a"), -6.10),
        step=0.01,
        format="%.2f",
    )

with col2:
    B_peil = st.number_input(
        "B-peil (min) [m NAP]",
        value=_as_float(extra.get("meetniveau_abba_b"), -6.20),
        step=0.01,
        format="%.2f",
    )

with col3:
    opp = st.number_input(
        "Oppervlakte [m²]",
        value=_as_float(extra.get("oppervlakte"), 0.0),
        min_value=0.0,
        step=0.001,
        format="%.3f",
    )
# ---------- ABBA-berekening (bit 9 / 9) ----------
BIT_INDEX_P1 = 9
BIT_INDEX_P2 = 9
with st.spinner("ABBA-debieten (noneq) berekenen…"):
    df_out = abba_berekenen_noneq(
        df_h=df_h, df_p1=df_p1, df_p2=df_p2 if not df_p2.empty else None,
        A_peil=A_peil, B_peil=B_peil, opp=opp,
        start_utc=start_utc, end_utc=end_utc,
        bit_index_p1=int(BIT_INDEX_P1), bit_index_p2=int(BIT_INDEX_P2),
    )

if df_out.empty:
    st.warning("Geen resultaten (controleer inputreeksen, periode of metadata).")
    st.stop()

# ---------- Neerslag (precip_cum) ----------
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
    precip_cum = None

# ---------- Result voor plot ----------
plot_df = df_out.copy()
plot_df["time_local"] = _to_local(plot_df["time"])
# pompreeksen positief tekenen (alleen visual)
plot_df["debiet_pomp_P1_pos"] = -1.0 * plot_df.get("debiet_pomp_P1", pd.Series(index=plot_df.index, dtype=float)).fillna(0.0)
plot_df["debiet_pomp_P2_pos"] = -1.0 * plot_df.get("debiet_pomp_P2", pd.Series(index=plot_df.index, dtype=float)).fillna(0.0)

# ---------- KPI's: draaiuren, schakelingen, debiet/draaiuren ----------
def _kpi_runtime_switches_and_ratio(df: pd.DataFrame, bedrijf_col: str, pump_col: str) -> Tuple[float,int,float]:
    """Return (draaiuren, schakelingen, debiet/draaiuren). debiet/draaiuren = totaal volume / draaiuren."""
    if df.empty or bedrijf_col not in df or "time" not in df or pump_col not in df:
        return 0.0, 0, 0.0
    d = df[["time", bedrijf_col, pump_col]].copy()
    d = d.sort_values("time")
    t = pd.to_datetime(d["time"], utc=True)
    dt = t.diff().dt.total_seconds().fillna(0.0)  # s
    # draaiuren: som( bedrijf * dt ) / 3600
    bedrijf = pd.to_numeric(d[bedrijf_col], errors="coerce").fillna(0.0)
    runtime_h = float((bedrijf * dt).sum() / 3600.0)
    # schakelingen: count van 0->1 overgangen
    # NB: rond bedrijf naar 0/1, dan diff>0 is een aan overgang
    b01 = bedrijf.round(0).astype(int)
    switches = int(((b01.diff().fillna(0) > 0).sum()))
    # totaal verpompt volume (m3): integraal van debiet_pomp (negatief) over tijd
    q = pd.to_numeric(d[pump_col], errors="coerce").fillna(0.0)  # m3/h (negatief bij uitstroom)
    volume_m3 = float((-1.0 * (q * dt / 3600.0)).sum())  # positief volume
    ratio = float(volume_m3 / runtime_h) if runtime_h > 0 else 0.0  # m3/uur
    return runtime_h, switches, ratio

p1_hours, p1_sw, p1_ratio = _kpi_runtime_switches_and_ratio(df_out, "bedrijf_P1", "debiet_pomp_P1")
p2_hours, p2_sw, p2_ratio = _kpi_runtime_switches_and_ratio(df_out, "bedrijf_P2", "debiet_pomp_P2")

# Toon als 3 boxen (totaal over beide pompen)
tot_hours = p1_hours + p2_hours
tot_sw    = p1_sw + p2_sw
# ratio als gewogen gemiddelde op volume? We nemen volume totaal / uren totaal:
# Hiervan reeds in helper: volume per pomp / uren pomp; nu voor totaal:
def _total_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    d = df.copy().sort_values("time")
    t = pd.to_datetime(d["time"], utc=True)
    dt = t.diff().dt.total_seconds().fillna(0.0)
    q_tot = (d.get("debiet_pomp_P1", 0.0).fillna(0.0) + d.get("debiet_pomp_P2", 0.0).fillna(0.0))
    # volume positief
    volume_tot = float((-1.0 * (q_tot * dt / 3600.0)).sum())
    # draaiuren totaal = som(bedrijf_P1 + bedrijf_P2 > 0 ?) of som beide afzonderlijk?
    # Jij vroeg "totaal draaiuren (pomp1 + pomp2 opgeteld)" -> som van individuele draaiuren
    hours_tot = tot_hours
    return float(volume_tot / hours_tot) if hours_tot > 0 else 0.0

ratio_tot = _total_ratio(df_out)

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""
        <div style="border:1px solid #DDD;border-radius:14px;padding:16px;text-align:center;background:#F8FAFF">
            <div style="font-size:13px;color:#666">Totaal draaiuren (P1+P2)</div>
            <div style="font-size:28px;color:#696;font-weight:700;margin-top:6px">{tot_hours:.2f} uur</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div style="border:1px solid #DDD;border-radius:14px;padding:16px;text-align:center;background:#F8FAFF">
            <div style="font-size:13px;color:#666">Totaal schakelingen (P1+P2)</div>
            <div style="font-size:28px;color:#696;font-weight:700;margin-top:6px">{tot_sw:d}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div style="border:1px solid #DDD;border-radius:14px;padding:16px;text-align:center;background:#F8FAFF">
            <div style="font-size:13px;color:#666">Debiet / draaiuren (gemiddeld)</div>
            <div style="font-size:28px;color:#696;font-weight:700;margin-top:6px">{ratio_tot:.1f} m³/uur</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("---")

# ---------- Sidebar: drempels (clip alléén de debiet-lijn) ----------
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

# ---------- Plotfuncties ----------
def plot_bedrijf(df: pd.DataFrame, height: int = 240) -> alt.Chart:
    want = {"bedrijf_p1", "bedrijf_p2"}
    cols = [c for c in df.columns if c.lower() in want]
    if not cols:
        return alt.Chart(pd.DataFrame({"time_local": [], "waarde": []})).mark_line().properties(height=height)
    b = df[["time"] + cols].copy()
    for c in cols:
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0).round(0)

    SCALE = 1000.0  # alleen visual
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

# ---------- Export naar Excel ----------
def export_to_excel(prefix: str,
                    df_h: pd.DataFrame | None,
                    df_p1: pd.DataFrame | None,
                    df_p2: pd.DataFrame | None,
                    df_out: pd.DataFrame | None,
                    plot_df: pd.DataFrame | None,
                    kpis: dict) -> bytes:
    import io
    import pandas as pd
    import numpy as np

    def _safe_df(x: pd.DataFrame | None) -> pd.DataFrame:
        return x if isinstance(x, pd.DataFrame) and not x.empty else pd.DataFrame()

    def _detz(df: pd.DataFrame) -> pd.DataFrame:
        """Maak alle datetime-kolommen tz-vrij (naive); werkt ook met datetime64[ns, tz]."""
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        for c in d.columns:
            s = d[c]
            # tz-aware datetime64?
            if is_datetime64tz_dtype(s):
                d[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
            # naive datetime64 -> laten staan
            elif is_datetime64_any_dtype(s):
                continue
            # soms zitten er Timestamp-objects in object-kolommen
            elif s.dtype == object:
                try:
                    s2 = pd.to_datetime(s, utc=True, errors="raise")
                    if is_datetime64tz_dtype(s2):
                        d[c] = s2.dt.tz_convert("UTC").dt.tz_localize(None)
                    elif is_datetime64_any_dtype(s2):
                        d[c] = s2  # al naive
                except Exception:
                    pass
        return d

    # Prepare frames (tz-vrij)
    h   = _detz(_safe_df(df_h))
    p1  = _detz(_safe_df(df_p1))
    p2  = _detz(_safe_df(df_p2))
    out = _detz(_safe_df(df_out))
    pl  = _detz(_safe_df(plot_df))

    # Meta netjes als strings zonder tz
    def _fmt(ts):
        try:
            return pd.to_datetime(ts).tz_convert("Europe/Amsterdam").tz_localize(None)
        except Exception:
            try:
                return pd.to_datetime(ts).tz_localize(None)
            except Exception:
                return str(ts)

    meta = pd.DataFrame({
        "prefix": [prefix],
        "periode_start": [_fmt(start_utc)],
        "periode_eind":  [_fmt(end_utc)],
        "A_peil": [A_peil],
        "B_peil": [B_peil],
        "opp_m2": [opp],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        h.to_excel(writer,  index=False, sheet_name="input_waterhoogte")
        p1.to_excel(writer, index=False, sheet_name="input_status_p1")
        p2.to_excel(writer, index=False, sheet_name="input_status_p2")
        out.to_excel(writer, index=False, sheet_name="output_reken")
        pl.to_excel(writer,  index=False, sheet_name="output_plot")
        pd.DataFrame([kpis]).to_excel(writer, index=False, sheet_name="KPIs")
        meta.to_excel(writer, index=False, sheet_name="meta")
    return buf.getvalue()


kpis = {
    "prefix": CODE_PREFIX,
    "periode": f"{start_utc} — {end_utc}",
    "P1_draaiuren_uur": round(p1_hours, 2),
    "P1_schakelingen": int(p1_sw),
    "P1_debiet_per_uur_m3ph": round(p1_ratio, 1),
    "P2_draaiuren_uur": round(p2_hours, 2),
    "P2_schakelingen": int(p2_sw),
    "P2_debiet_per_uur_m3ph": round(p2_ratio, 1),
    "Totaal_draaiuren_uur": round(tot_hours, 2),
    "Totaal_schakelingen": int(tot_sw),
    "Totaal_debiet_per_uur_m3ph": round(ratio_tot, 1),
}

xls_bytes = export_to_excel(CODE_PREFIX, df_h, df_p1, df_p2, df_out, plot_df, kpis)
st.download_button(
    "⬇️ Exporteer naar Excel",
    data=xls_bytes,
    file_name=f"abba_{CODE_PREFIX}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
