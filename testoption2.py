import streamlit as st
import datetime as dt
import pandas as pd
import requests
import numpy as np
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, percentileofscore
import math
from scipy.interpolate import CubicSpline

###########################################
# Global settings and constants
###########################################
CONTRACT_SIZE = 1  # Adjust this to the appropriate contract size

###########################################
# Thalex API details
###########################################
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"  # Endpoint for fetching available instruments
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"

# Dictionary for rolling window configuration
windows = {"7D": "vrp_7d"}
DEFAULT_EXPIRY_STR = "28MAR25"

# Calculate time to expiry using default expiry
expiry_date = dt.datetime.strptime(DEFAULT_EXPIRY_STR, "%d%b%y")
current_date = dt.datetime.now()
days_to_expiry = (expiry_date - current_date).days
T_YEARS = days_to_expiry / 365

###########################################
# Helper functions for API parameters
###########################################
def params(instrument_name):
    now = dt.datetime.now()
    start_dt = now - dt.timedelta(days=7)
    return {
        "from": int(start_dt.timestamp()),
        "to": int(now.timestamp()),
        "resolution": "5m",
        "instrument_name": instrument_name,
    }

COLUMNS = [
    "ts",
    "mark_price_open",
    "mark_price_high",
    "mark_price_low",
    "mark_price_close",
    "iv_open",
    "iv_high",
    "iv_low",
    "iv_close",
]

###########################################
# Expiration Date Helper Functions
###########################################
def get_valid_expiration_options(current_date=None):
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < 14:
        return [14, 28]
    elif current_date.day < 28:
        return [28]
    else:
        return [14, 28]

def compute_expiry_date(selected_day, current_date=None):
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < selected_day:
        try:
            expiry = current_date.replace(day=selected_day, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            st.error("Invalid expiration date for current month.")
            return None
    else:
        year = current_date.year + (current_date.month // 12)
        month = (current_date.month % 12) + 1
        try:
            expiry = dt.datetime(year, month, selected_day)
        except ValueError:
            st.error("Invalid expiration date for next month.")
            return None
    return expiry

###########################################
# Credentials & Login Functions
###########################################
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user:
            usernames = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            passwords = [line.strip() for line in f_pass if line.strip()]
        if len(usernames) != len(passwords):
            st.error("The number of usernames and passwords do not match.")
            return {}
        creds = dict(zip(usernames, passwords))
        return creds
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Please Log In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            creds = load_credentials()
            if username in creds and creds[username] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# Instruments Fetching & Filtering Functions
###########################################
def fetch_instruments():
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type, expiry_str):
    filtered = [
        inst["instrument_name"] for inst in instruments
        if inst["instrument_name"].startswith(f"BTC-{expiry_str}") and inst["instrument_name"].endswith(f"-{option_type}")
    ]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    response = requests.get(url_mark_price, params=params(instrument_name))
    if response.status_code != 200:
        return None
    data = response.json()
    marks = get_in(["result", "mark"])(data)
    if not marks:
        return None
    df = pd.DataFrame(marks, columns=COLUMNS)
    df = df.sort_values("ts")
    return df["iv_close"].iloc[-1]

def get_filtered_instruments(spot_price, expiry_str=DEFAULT_EXPIRY_STR, t_years=T_YEARS, multiplier=1):
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    if not calls_all:
        raise Exception(f"No call instruments found for expiry {expiry_str}.")
    strike_list = []
    for inst in calls_all:
        parts = inst.split("-")
        if len(parts) >= 3 and parts[2].isdigit():
            strike_list.append((inst, int(parts[2])))
    if not strike_list:
        raise Exception(f"No valid call instruments with strikes found for expiry {expiry_str}.")
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    if not strikes:
        raise Exception("No strikes available for filtering.")
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument.")
    lower_bound = spot_price * np.exp(-actual_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(actual_iv * np.sqrt(t_years) * multiplier)
    filtered_calls = [inst for inst in calls_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_puts = [inst for inst in puts_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_calls.sort(key=lambda x: int(x.split("-")[2]))
    filtered_puts.sort(key=lambda x: int(x.split("-")[2]))
    if not filtered_calls and not filtered_puts:
        raise Exception("No instruments left after applying the theoretical range filter.")
    return filtered_calls, filtered_puts

###########################################
# Data Fetching Functions
###########################################
@st.cache_data(ttl=30)
def fetch_data(instruments_tuple):
    instruments = list(instruments_tuple)
    df = (
        pipe(
            {name: requests.get(url_mark_price, params=params(name)) for name in instruments},
            valmap(requests.Response.json),
            valmap(get_in(["result", "mark"])),
            valmap(curry(pd.DataFrame, columns=COLUMNS)),
            valfilter(lambda df: not df.empty),
            pd.concat,
        )
        .droplevel(1)
        .reset_index(names=["instrument_name"])
        .assign(date_time=lambda df: pd.to_datetime(df["ts"], unit="s")
                .dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York"))
        .assign(k=lambda df: df["instrument_name"].map(lambda s: int(s.split("-")[2])
                                                      if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
        .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
    )
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    params = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

###########################################
# Updated Kraken Data Fetch (Dual Timeframe)
###########################################
def fetch_kraken_data():
    kraken = ccxt.kraken()
    now_dt = dt.datetime.now()
    start_dt = now_dt - dt.timedelta(days=365)
    ohlcv_5m = kraken.fetch_ohlcv("BTC/USD", timeframe="5m",
                                   since=int(start_dt.timestamp()) * 1000,
                                   limit=3000)
    ohlcv_1d = kraken.fetch_ohlcv("BTC/USD", timeframe="1d",
                                   since=int(start_dt.timestamp()) * 1000)
    df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_5m["date_time"] = pd.to_datetime(df_5m["timestamp"], unit="ms")
    df_5m = df_5m.set_index("date_time")
    df_1d = pd.DataFrame(ohlcv_1d, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_1d["date_time"] = pd.to_datetime(df_1d["timestamp"], unit="ms")
    df_1d = df_1d.set_index("date_time")
    full_df = pd.concat([df_5m, df_1d], axis=0).sort_index()
    return full_df[~full_df.index.duplicated()]

###########################################
# Parkinson Volatility Calculation
###########################################
def calculate_parkinson_volatility(price_data, window_days=7, annualize_days=365):
    """
    Calculate Parkinson volatility using daily high/low prices.
    """
    if price_data.empty or not {'high', 'low'}.issubset(price_data.columns):
        return np.nan
    if "date_time" in price_data.columns:
        daily_data = price_data.resample('D', on='date_time').agg({'high': 'max', 'low': 'min'}).dropna()
    else:
        daily_data = price_data.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    if len(daily_data) < window_days:
        return np.nan
    daily_data['log_hl_ratio'] = np.log(daily_data['high'] / daily_data['low'])
    parkinson_var = daily_data['log_hl_ratio'].rolling(window=window_days, min_periods=1).apply(
        lambda x: np.sum(x**2) / (4 * np.log(2) * window_days), raw=True)
    annualized_vol = np.sqrt(parkinson_var * annualize_days)
    if "date_time" in price_data.columns:
        return annualized_vol.reindex(price_data.index, method='ffill')
    else:
        return annualized_vol.reindex(price_data.index, method='ffill')

def compute_daily_realized_volatility(df, window_days=7):
    if 'date_time' in df.columns:
        df_daily = df.resample('D', on='date_time').agg({'high': 'max', 'low': 'min'}).dropna()
    else:
        df_daily = df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    daily_vols = []
    for i in range(window_days, len(df_daily) + 1):
        window = df_daily.iloc[max(0, i - window_days):i]
        if len(window) < window_days:
            continue
        vol = calculate_parkinson_volatility(window, window_days=window_days)
        daily_vols.append(vol.iloc[-1] if not vol.empty else np.nan)
    return [v for v in daily_vols if not np.isnan(v)]
    
def compute_daily_average_iv(df_iv_agg):
    daily_iv = df_iv_agg["iv_mean"].resample("D").mean(numeric_only=True).dropna().tolist()
    return daily_iv

def compute_historical_vrp(daily_iv, daily_rv):
    n = min(len(daily_iv), len(daily_rv))
    return [(iv ** 2) - (rv ** 2) for iv, rv in zip(daily_iv[:n], daily_rv[:n])]

###########################################
# Option Delta Calculation Function
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365.25 * 86400)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    if sigma <= 0:
        return np.nan
    try:
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

###########################################
# Delta-Based Dynamic Regime Functions
###########################################
def compute_average_delta(df_calls, df_puts, S):
    if "delta" not in df_calls.columns:
        df_calls["delta"] = df_calls.apply(lambda row: compute_delta(row, S), axis=1)
    if "delta" not in df_puts.columns:
        df_puts["delta"] = df_puts.apply(lambda row: compute_delta(row, S), axis=1)
    df_calls_mean = df_calls.groupby("date_time", as_index=False)["delta"].mean().rename(columns={"delta": "call_delta_avg"})
    df_puts_mean = df_puts.groupby("date_time", as_index=False)["delta"].mean().rename(columns={"delta": "put_delta_avg"})
    df_merged = pd.merge(df_calls_mean, df_puts_mean, on="date_time", how="outer").sort_values("date_time")
    df_merged["delta_diff"] = df_merged["call_delta_avg"] - df_merged["put_delta_avg"]
    return df_merged

def rolling_percentile_zones(df, column="delta_diff", window="1D", lower_percentile=33, upper_percentile=66):
    df = df.set_index("date_time").sort_index()
    def percentile_in_window(x, q):
        return np.percentile(x, q)
    df["rolling_lower_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, lower_percentile), raw=False)
    df["rolling_upper_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, upper_percentile), raw=False)
    return df.reset_index()

def classify_regime(row):
    if pd.isna(row["rolling_lower_zone"]) or pd.isna(row["rolling_upper_zone"]):
        return "Neutral"
    if row["delta_diff"] > row["rolling_upper_zone"]:
        return "Risk-On"
    elif row["delta_diff"] < row["rolling_lower_zone"]:
        return "Risk-Off"
    else:
        return "Neutral"

###########################################
# Gamma Calculation Function
###########################################
def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    if sigma <= 0:
        return np.nan
    try:
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

###########################################
# Gamma Exposure (GEX) Calculation Function
###########################################
def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    # Standard GEX = Gamma * Spot^2 * OpenInterest * Contract Size
    return gamma * oi * (S ** 2) * CONTRACT_SIZE

###########################################
# Gamma & GEX Visualizations
###########################################
def plot_gamma_heatmap(df):
    st.subheader("Gamma Heatmap by Strike and Time")
    fig_gamma_heatmap = px.density_heatmap(
        df,
        x="date_time",
        y="k",
        z="gamma",
        color_continuous_scale="Viridis",
        title="Gamma by Strike Over Time"
    )
    fig_gamma_heatmap.update_layout(height=400, width=800)
    st.plotly_chart(fig_gamma_heatmap, use_container_width=True)

def plot_gex_by_strike(df_gex):
    st.subheader("Gamma Exposure (GEX) by Strike")
    fig_gex = px.bar(
        df_gex,
        x="strike",
        y="gex",
        color="option_type",
        title="Gamma Exposure (GEX) by Strike",
        labels={"gex": "GEX", "strike": "Strike Price"}
    )
    fig_gex.update_layout(height=400, width=800)
    st.plotly_chart(fig_gex, use_container_width=True)

def plot_net_gex(df_gex, spot_price):
    st.subheader("Net Gamma Exposure by Strike")
    df_gex_net = df_gex.groupby("strike").apply(
        lambda x: x.loc[x["option_type"] == "C", "gex"].sum() - x.loc[x["option_type"] == "P", "gex"].sum()
    ).reset_index(name="net_gex")
    df_gex_net["sign"] = df_gex_net["net_gex"].apply(lambda val: "Negative" if val < 0 else "Positive")
    fig_net_gex = px.bar(
        df_gex_net,
        x="strike",
        y="net_gex",
        color="sign",
        color_discrete_map={"Negative": "orange", "Positive": "blue"},
        title="Net Gamma Exposure (Calls GEX - Puts GEX)",
        labels={"net_gex": "Net GEX", "strike": "Strike Price"}
    )
    fig_net_gex.add_hline(y=0, line_dash="dash", line_color="red")
    fig_net_gex.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="lightgrey",
        annotation_text=f"Spot {spot_price:.0f}",
        annotation_position="top right"
    )
    fig_net_gex.update_layout(height=400, width=800)
    st.plotly_chart(fig_net_gex, use_container_width=True)

###########################################
# Percentile-Based Risk Classifications
###########################################
def classify_volatility_regime(current_vol, historical_vols):
    percentile = percentileofscore(historical_vols, current_vol)
    if percentile < 5:
        return "Low Volatility"
    elif percentile > 95:
        return "High Volatility"
    else:
        return "Medium Volatility"

def classify_vrp_regime(current_vrp, historical_vrps):
    percentile = percentileofscore(historical_vrps, current_vrp)
    if current_vrp < 0:
        return "Long Volatility"
    elif percentile > 75:
        return "Short Volatility"
    elif percentile < 25:
        return "Long Volatility"
    else:
        return "Neutral"

###########################################
# EV Calculation Functions for Different Strategies
###########################################
def calculate_atm_straddle_ev(ticker_list, spot_price, T, rv):
    # For ATM Straddle EV analysis using variance difference formula, expressed as a percentage
    tolerance = spot_price * 0.02  
    atm_candidates = [item for item in ticker_list if abs(item["strike"] - spot_price) <= tolerance]
    if not atm_candidates:
        return None
    atm_strikes = {}
    for item in atm_candidates:
        strike = item["strike"]
        if strike not in atm_strikes:
            atm_strikes[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            atm_strikes[strike]["iv_sum"] += item["iv"]
            atm_strikes[strike]["count"] += 1
    ev_candidates = []
    for strike, data in atm_strikes.items():
        avg_iv = data["iv_sum"] / data["count"]
        # EV using variance difference formula: EV = ((IV^2 - RV^2) * T)/2, then converted to percentage
        ev_value = (((avg_iv**2 - rv**2) * T) / 2) * 100
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV (%)": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV (%)", ascending=False)

def calculate_limited_otm_put_ev(ticker_list, spot_price, T, rv):
    # For Limited OTM Puts: consider put options with strike below spot.
    tolerance = spot_price * 0.10  
    candidates = [item for item in ticker_list if item["option_type"] == "P" and item["strike"] < spot_price and (spot_price - item["strike"]) <= tolerance]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = (((avg_iv**2 - rv**2) * T) / 2) * 100
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV (%)": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV (%)", ascending=False)

def calculate_call_spread_ev(ticker_list, spot_price, T, rv):
    # For Call Spreads: consider call options with strike above spot.
    tolerance = spot_price * 0.10  
    candidates = [item for item in ticker_list if item["option_type"] == "C" and item["strike"] > spot_price and (item["strike"] - spot_price) <= tolerance]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = (((avg_iv**2 - rv**2) * T) / 2) * 100
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV (%)": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV (%)", ascending=False)

def calculate_strangle_ev(ticker_list, spot_price, T, rv):
    # For Strangles: include OTM calls and puts.
    tolerance = spot_price * 0.10  
    candidates = [item for item in ticker_list if ((item["option_type"] == "C" and item["strike"] > spot_price and (item["strike"] - spot_price) <= tolerance)
                                                     or (item["option_type"] == "P" and item["strike"] < spot_price and (spot_price - item["strike"]) <= tolerance))]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = (((avg_iv**2 - rv**2) * T) / 2) * 100
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV (%)": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV (%)", ascending=False)

def calculate_naked_call_ev(ticker_list, spot_price, T, rv):
    # For Naked Calls: consider all call options with strike above spot.
    candidates = [item for item in ticker_list if item["option_type"] == "C" and item["strike"] > spot_price]
    if not candidates:
        return None
    group = {}
    for item in candidates:
        strike = item["strike"]
        if strike not in group:
            group[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            group[strike]["iv_sum"] += item["iv"]
            group[strike]["count"] += 1
    ev_candidates = []
    for strike, data in group.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = (((avg_iv**2 - rv**2) * T) / 2) * 100
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV (%)": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV (%)", ascending=False)

def calculate_small_atm_straddle_ev(ticker_list, spot_price, T, rv):
    # For Small ATM Straddles, we reuse the ATM straddle EV analysis.
    return calculate_atm_straddle_ev(ticker_list, spot_price, T, rv)

###########################################
# Dynamic Volatility Adjustment Using Volatility Smile
###########################################
def adjust_volatility_with_smile(strike, smile_df):
    """
    Adjust the implied volatility for a given strike based on the observed volatility smile.
    The smile_df DataFrame must contain 'strike' and 'iv' columns.
    """
    sorted_smile = smile_df.sort_values("strike")
    strikes = sorted_smile["strike"].values
    ivs = sorted_smile["iv"].values
    adjusted_iv = np.interp(strike, strikes, ivs)
    return adjusted_iv

def interpolate_iv(strikes, ivs, strike):
    if not isinstance(strikes, (np.ndarray, list)) or not isinstance(ivs, (np.ndarray, list)):
        raise ValueError("strikes and ivs must be arrays or lists.")
    if len(strikes) != len(ivs):
        raise ValueError("strikes and ivs must have the same length.")
    if len(strikes) < 4:
        raise ValueError("At least 4 points are required for cubic spline interpolation.")
    valid_mask = ~np.isnan(strikes) & ~np.isnan(ivs) & ~np.isinf(strikes) & ~np.isinf(ivs)
    strikes_clean = np.array(strikes)[valid_mask]
    ivs_clean = np.array(ivs)[valid_mask]
    if len(strikes_clean) < 4:
        raise ValueError("At least 4 valid points are required for cubic spline interpolation.")
    sort_idx = np.argsort(strikes_clean)
    strikes_sorted = strikes_clean[sort_idx]
    ivs_sorted = ivs_clean[sort_idx]
    cs = CubicSpline(strikes_sorted, ivs_sorted)
    if strike < strikes_sorted.min():
        adjusted_iv = ivs_sorted[0]
    elif strike > strikes_sorted.max():
        adjusted_iv = ivs_sorted[-1]
    else:
        adjusted_iv = cs(strike)
    return adjusted_iv

###########################################
# Build Smile DataFrame from Ticker List
###########################################
def build_smile_df(ticker_list):
    df = pd.DataFrame(ticker_list)
    df = df.dropna(subset=["iv"])
    smile_df = df.groupby("strike", as_index=False)["iv"].mean()
    return smile_df

###########################################
# Ticker List Builder with Smile Adjustment
###########################################
def build_ticker_list(all_instruments, spot, T, smile_df):
    ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not (ticker_data and "open_interest" in ticker_data):
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        raw_iv = ticker_data.get("iv", None)
        if raw_iv is None:
            continue
        # Adjust IV using the observed volatility smile data
        adjusted_iv = adjust_volatility_with_smile(strike, smile_df)
        try:
            d1 = (np.log(spot / strike) + 0.5 * adjusted_iv**2 * T) / (adjusted_iv * np.sqrt(T))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "delta": delta_est,
            "iv": adjusted_iv
        })
    return ticker_list

###########################################
# Trade Strategy Evaluation Function
###########################################
def evaluate_trade_strategy(df, spot_price, risk_tolerance="Moderate", df_iv_agg_reset=None,
                            historical_vols=None, historical_vrps=None, days_to_expiration=7):
    rv_vol_series = calculate_parkinson_volatility(df_kraken, window_days=7)
    rv_vol = rv_vol_series.iloc[-1] if not rv_vol_series.empty else np.nan
    iv_vol = df["iv_close"].mean() if not df.empty else np.nan

    rv_var = rv_vol ** 2 if not pd.isna(rv_vol) else np.nan
    iv_var = iv_vol ** 2 if not pd.isna(iv_vol) else np.nan
    current_vrp = iv_var - rv_var if (not pd.isna(iv_vol) and not pd.isna(rv_vol)) else np.nan

    divergence = abs(iv_vol - rv_vol) / rv_vol if (rv_vol != 0 and not pd.isna(iv_vol)) else np.nan
    if not np.isnan(divergence) and divergence > 0.20:
        st.write(f"Alert: IV and RV diverge by {divergence*100:.2f}% (threshold: 20%).")

    # Compute rolling 1-day IV mean and std for market regime determination
    df_iv_agg = df.groupby("date_time")["iv_close"].mean().to_frame(name="iv_mean")
    df_iv_agg.index = pd.to_datetime(df_iv_agg.index)
    df_iv_agg = df_iv_agg.resample("5min").mean().ffill().sort_index()
    df_iv_agg["iv_rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()
    df_iv_agg["iv_rolling_std"] = df_iv_agg["iv_mean"].rolling("1D").std()
    df_iv_agg["upper_zone"] = df_iv_agg["iv_rolling_mean"] + df_iv_agg["iv_rolling_std"]
    df_iv_agg["lower_zone"] = df_iv_agg["iv_rolling_mean"] - df_iv_agg["iv_rolling_std"]
    latest_iv = df_iv_agg["iv_mean"].iloc[-1]
    latest_upper = df_iv_agg["upper_zone"].iloc[-1]
    latest_lower = df_iv_agg["lower_zone"].iloc[-1]
    if latest_iv > latest_upper:
        market_regime = "Risk-Off"
    elif latest_iv < latest_lower:
        market_regime = "Risk-On"
    else:
        market_regime = "Neutral"
    vol_regime = market_regime
    vrp_regime = classify_vrp_regime(current_vrp, historical_vrps) if historical_vrps and len(historical_vrps) > 0 else "Neutral"

    call_items = [item for item in ticker_list if item["option_type"] == "C"]
    put_items = [item for item in ticker_list if item["option_type"] == "P"]
    call_oi_total = sum(item["open_interest"] for item in call_items)
    put_oi_total = sum(item["open_interest"] for item in put_items)
    avg_call_delta = (sum(item["delta"] * item["open_interest"] for item in call_items) / call_oi_total) if call_oi_total > 0 else 0
    avg_put_delta = (sum(item["delta"] * item["open_interest"] for item in put_items) / put_oi_total) if put_oi_total > 0 else 0

    df_ticker = pd.DataFrame(ticker_list) if ticker_list else pd.DataFrame()
    call_oi = df_ticker[df_ticker["option_type"] == "C"]["open_interest"].sum() if not df_ticker.empty else 0
    put_oi = df_ticker[df_ticker["option_type"] == "P"]["open_interest"].sum() if not df_ticker.empty else 0
    put_call_ratio = put_oi / call_oi if call_oi > 0 else np.inf

    df_calls = df[df["option_type"] == "C"].copy()
    df_puts = df[df["option_type"] == "P"].copy()
    if "gamma" not in df_calls.columns:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    if "gamma" not in df_puts.columns:
        df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    avg_call_gamma = df_calls["gamma"].mean() if not df_calls.empty else 0
    avg_put_gamma = df_puts["gamma"].mean() if not df_puts.empty else 0

    # Trading Recommendation Logic
    if vrp_regime == "Long Volatility":
        if risk_tolerance == "Conservative":
            recommendation = "Long Volatility (Conservative): Buy limited OTM Puts"
            position = "Limited OTM Puts"
            hedge_action = "Hedge lightly with BTC futures short"
        elif risk_tolerance == "Moderate":
            recommendation = "Long Volatility (Neutral): Consider delta-hedged straddles"
            position = "ATM Straddle"
            hedge_action = "Implement dynamic delta hedging"
        else:
            recommendation = "Long Volatility (Aggressive): Take leveraged long volatility positions"
            position = "Leveraged Long Straddle"
            hedge_action = "Aggressively hedge using BTC futures"
    elif vrp_regime == "Short Volatility":
        if risk_tolerance == "Conservative":
            recommendation = "Short Volatility (Conservative): Sell a small number of call spreads"
            position = "Call Spread"
            hedge_action = "Hedge by buying BTC futures"
        elif risk_tolerance == "Moderate":
            recommendation = "Short Volatility (Neutral): Sell strangles with tight stops"
            position = "Strangle"
            hedge_action = "Monitor and adjust hedge dynamically"
        else:
            recommendation = "Short Volatility (Aggressive): Consider naked call selling"
            position = "Naked Calls"
            hedge_action = "Aggressively hedge with BTC futures"
    else:
        recommendation = "Gamma Scalping (Neutral): Consider buying small ATM straddles"
        position = "Small ATM Straddle"
        hedge_action = "Implement light delta hedging"

    return {
        "recommendation": recommendation,
        "position": position,
        "hedge_action": hedge_action,
        "iv": iv_vol,
        "rv": rv_vol,
        "vol_regime": vol_regime,
        "vrp_regime": vrp_regime,
        "put_call_ratio": put_call_ratio,
        "avg_call_delta": avg_call_delta,
        "avg_put_delta": avg_put_delta,
        "avg_call_gamma": avg_call_gamma,
        "avg_put_gamma": avg_put_gamma
    }

###########################################
# Main Dashboard
###########################################
def main():
    login()
    st.title("Crypto Options Visualization Dashboard with Dynamic Volatility Adjustments")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    current_date = dt.datetime.now()
    valid_days = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_days)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    global df_kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts

    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return

    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")

    df_iv_agg = df.groupby("date_time", as_index=False)["iv_close"].mean().rename(columns={"iv_close": "iv_mean"})
    df_iv_agg["date_time"] = pd.to_datetime(df_iv_agg["date_time"])
    df_iv_agg = df_iv_agg.set_index("date_time")
    df_iv_agg = df_iv_agg.resample("5min").mean().ffill().sort_index()
    # Compute rolling IV statistics for market regime determination
    df_iv_agg["iv_rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()
    df_iv_agg["iv_rolling_std"] = df_iv_agg["iv_mean"].rolling("1D").std()
    df_iv_agg["upper_zone"] = df_iv_agg["iv_rolling_mean"] + df_iv_agg["iv_rolling_std"]
    df_iv_agg["lower_zone"] = df_iv_agg["iv_rolling_mean"] - df_iv_agg["iv_rolling_std"]
    df_iv_agg_reset = df_iv_agg.reset_index()

    # Build preliminary ticker list with raw IV values
    preliminary_ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not (ticker_data and "open_interest" in ticker_data):
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        raw_iv = ticker_data.get("iv", None)
        if raw_iv is None:
            continue
        preliminary_ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "iv": raw_iv
        })
    smile_df = build_smile_df(preliminary_ticker_list)

    # Rebuild ticker list using dynamic volatility adjustment based on the observed smile
    global ticker_list
    ticker_list = build_ticker_list(all_instruments, spot_price, T_YEARS, smile_df)

    daily_rv = compute_daily_realized_volatility(df_kraken)
    daily_iv = compute_daily_average_iv(df_iv_agg)
    historical_vrps = compute_historical_vrp(daily_iv, daily_rv)

    st.subheader("Volatility Trading Decision Tool")
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance",
                                          options=["Conservative", "Moderate", "Aggressive"],
                                          index=1)
    trade_decision = evaluate_trade_strategy(
        df, spot_price, risk_tolerance, df_iv_agg_reset,
        historical_vols=daily_rv,
        historical_vrps=historical_vrps,
        days_to_expiration=days_to_expiry
    )

    st.write("### Market and Volatility Metrics")
    st.write(f"Implied Volatility (IV): {trade_decision['iv']:.2%}")
    st.write(f"Realized Volatility (RV): {trade_decision['rv']:.2%}")
    st.write(f"Market Regime: {trade_decision['vol_regime']}")
    st.write(f"VRP Regime: {trade_decision['vrp_regime']}")
    st.write(f"Put/Call Open Interest Ratio: {trade_decision['put_call_ratio']:.2f}")
    st.write(f"Average Call Delta: {trade_decision['avg_call_delta']:.4f}")
    st.write(f"Average Put Delta: {trade_decision['avg_put_delta']:.4f}")
    st.write(f"Average Gamma: {trade_decision['avg_call_gamma']:.6f}")

    st.subheader("Trading Recommendation")
    st.write(f"**Recommendation:** {trade_decision['recommendation']}")
    st.write(f"**Position:** {trade_decision['position']}")
    st.write(f"**Hedge Action:** {trade_decision['hedge_action']}")

    # EV Analysis: Use the appropriate EV function based on recommended position.
    rv_series = calculate_parkinson_volatility(df_kraken, window_days=7, annualize_days=365)
    rv_scalar = rv_series.iloc[-1] if not rv_series.empty else np.nan
    position = trade_decision['position']
    if position in ["ATM Straddle", "Leveraged Long Straddle"]:
        df_ev = calculate_atm_straddle_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
        st.subheader("ATM Straddle EV Analysis")
    elif position == "Limited OTM Puts":
        df_ev = calculate_limited_otm_put_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
        st.subheader("Limited OTM Put EV Analysis")
    elif position == "Call Spread":
        df_ev = calculate_call_spread_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
        st.subheader("Call Spread EV Analysis")
    elif position == "Strangle":
        df_ev = calculate_strangle_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
        st.subheader("Strangle EV Analysis")
    elif position == "Naked Calls":
        df_ev = calculate_naked_call_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
        st.subheader("Naked Call EV Analysis")
    elif position == "Small ATM Straddle":
        df_ev = calculate_small_atm_straddle_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
        st.subheader("Small ATM Straddle EV Analysis")
    else:
        df_ev = None
        st.subheader("EV Analysis")
        st.write("EV analysis for the selected position is not implemented yet.")

    if df_ev is not None and not df_ev.empty and not df_ev["EV (%)"].isna().all():
        df_ev_clean = df_ev.dropna(subset=["EV (%)"])
        if not df_ev_clean.empty:
            best_candidate = df_ev_clean.loc[df_ev_clean["EV (%)"].idxmax()]
            best_strike = best_candidate["Strike"]
            st.write("Candidate Strikes and their Expected Value (EV %) :")
            st.dataframe(df_ev_clean)
            st.write(f"Recommended Strike based on highest EV: {best_strike}")
        else:
            st.write("No candidates found with valid EV values.")
    else:
        st.write("No candidates found within tolerance for EV calculation.")

    if st.button("Simulate Trade"):
        st.write("Simulating trade based on recommendation...")
        st.write("Position Size: Adjust based on capital (e.g., 1-5% of portfolio for chosen risk tolerance)")
        st.write("Monitor price and volatility in real-time and adjust hedges dynamically.")

    if not df_calls.empty and not df_puts.empty:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
        df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    st.subheader("Volatility Smile at Latest Timestamp")
    latest_ts = df["date_time"].max()
    smile_df_latest = df[df["date_time"] == latest_ts]
    if not smile_df_latest.empty:
        atm_strike = smile_df_latest.loc[smile_df_latest["mark_price_close"].idxmax(), "k"]
        smile_df_latest = smile_df_latest.sort_values(by="k")
        fig_vol_smile = px.line(smile_df_latest, x="k", y="iv_close", markers=True,
                                title=f"Volatility Smile at {latest_ts.strftime('%d %b %H:%M')}",
                                labels={"iv_close": "IV", "k": "Strike"})
        cheap_hedge_strike = smile_df_latest.loc[smile_df_latest["iv_close"].idxmin(), "k"]
        fig_vol_smile.add_vline(x=cheap_hedge_strike, line=dict(dash="dash", color="green"),
                                annotation_text=f"Cheap Hedge ({cheap_hedge_strike})", annotation_position="top")
        fig_vol_smile.add_vline(x=spot_price, line=dict(dash="dash", color="blue"),
                                annotation_text=f"Price: {spot_price:.2f}", annotation_position="bottom left")
        fig_vol_smile.update_layout(height=400, width=600)
        st.plotly_chart(fig_vol_smile, use_container_width=True)
        plot_gamma_heatmap(pd.concat([df_calls, df_puts]))

    gex_data = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if ticker_data and "open_interest" in ticker_data:
            oi = ticker_data["open_interest"]
        else:
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        if option_type == "C":
            candidate = df_calls[df_calls["instrument_name"] == instrument]
        else:
            candidate = df_puts[df_puts["instrument_name"] == instrument]
        if candidate.empty:
            continue
        row = candidate.iloc[0]
        gex = compute_gex(row, spot_price, oi)
        gex_data.append({"strike": strike, "gex": gex, "option_type": option_type})
    df_gex = pd.DataFrame(gex_data)
    if not df_gex.empty:
        plot_gex_by_strike(df_gex)
        plot_net_gex(df_gex, spot_price)

if __name__ == '__main__':
    main()
