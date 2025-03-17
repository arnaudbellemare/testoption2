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
# EXPIRATION DATE SELECTION FUNCTIONS
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
# Thalex API details
###########################################
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
windows = {"7D": "vrp_7d"}

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
# CREDENTIALS & LOGIN FUNCTIONS
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
        return dict(zip(usernames, passwords))
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
                st.success("Logged in successfully! Click login again to open the app.")
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type, expiry_str):
    return sorted([inst["instrument_name"] for inst in instruments
                   if inst["instrument_name"].startswith(f"BTC-{expiry_str}") and inst["instrument_name"].endswith(f"-{option_type}")])

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

def get_filtered_instruments(spot_price, expiry_str, t_years, multiplier=1):
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    if not calls_all:
        raise Exception(f"No call instruments found for expiry {expiry_str}.")
    strike_list = [(inst, int(inst.split("-")[2])) for inst in calls_all]
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
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
    return filtered_calls, filtered_puts

###########################################
# DATA FETCHING FUNCTIONS
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
        .assign(k=lambda df: df["instrument_name"].map(lambda s: int(s.split("-")[2]) if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
        .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
    )
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    params_ = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params_)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

def fetch_kraken_data():
    kraken = ccxt.kraken()
    now_dt = dt.datetime.now()
    start_dt = now_dt - dt.timedelta(days=7)
    since = int(start_dt.timestamp() * 1000)
    ohlcv = kraken.fetch_ohlcv("BTC/USD", timeframe="5m", since=since, limit=3000)
    df_kraken = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df_kraken.empty:
        return pd.DataFrame()
    df_kraken["date_time"] = pd.to_datetime(df_kraken["timestamp"], unit="ms")
    df_kraken["date_time"] = df_kraken["date_time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df_kraken = df_kraken.sort_values(by="date_time").reset_index(drop=True)
    cutoff_start = (now_dt - dt.timedelta(days=7)).astimezone(df_kraken["date_time"].dt.tz)
    df_kraken = df_kraken[df_kraken["date_time"] >= cutoff_start]
    return df_kraken

###########################################
# REALIZED VOLATILITY FUNCTIONS (as provided)
###########################################
def calculate_ewma_roger_satchell_volatility(price_data, span):
    df = price_data.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    ewma_rs = df['rs'].ewm(span=span, adjust=False).mean()
    return np.sqrt(ewma_rs.clip(lower=0))

def compute_realized_volatility_5min(df, annualize_days=365):
    df = df.copy()
    df['rs'] = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))
    total_variance = df['rs'].sum()
    if total_variance <= 0:
        return 0.0
    N = len(df)
    M = annualize_days * 24 * 12
    annualization_factor = np.sqrt(M / N)
    return np.sqrt(total_variance) * annualization_factor

def calculate_btc_annualized_volatility_daily(df):
    if "date_time" not in df.columns:
        df = df.reset_index()
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    last_30_returns = df_daily["daily_return"].dropna().tail(30)
    if last_30_returns.empty:
        return np.nan
    daily_std = last_30_returns.std()
    return daily_std * np.sqrt(365)

def calculate_daily_realized_volatility_series(df):
    if "date_time" not in df.columns:
        df = df.reset_index()
    df_daily = df.set_index("date_time").resample("D").last().dropna(subset=["close"])
    df_daily["daily_return"] = df_daily["close"].pct_change()
    return df_daily["daily_return"].rolling(window=30).std() * np.sqrt(365)

###########################################
# OPTION DELTA & GAMMA FUNCTIONS
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
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
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def compute_gex(row, S, oi):
    gamma_val = compute_gamma(row, S)
    if gamma_val is None or np.isnan(gamma_val):
        return np.nan
    return gamma_val * oi * (S ** 2) * 0.01

###########################################
# EV & COMPOSITE SCORE FUNCTIONS
###########################################
def compute_ev(adjusted_iv, rv, T, position_side):
    if position_side.lower() == "short":
        return (((adjusted_iv**2 - rv**2) * T) / 2) * 100
    else:
        return (((rv**2 - adjusted_iv**2) * T) / 2) * 100

def compute_gamma_value(spot, strike, sigma, T):
    d1 = (np.log(spot/strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (spot * sigma * np.sqrt(T))

def adjust_volatility_with_smile(strike, smile_df):
    sorted_smile = smile_df.sort_values("strike")
    strikes = sorted_smile["strike"].values
    ivs = sorted_smile["iv"].values
    return np.interp(strike, strikes, ivs)

def build_ticker_list_with_metrics(all_instruments, spot, T, smile_df, rv, position_side="short"):
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
        adjusted_iv = adjust_volatility_with_smile(strike, smile_df)
        try:
            d1 = (np.log(spot / strike) + 0.5 * adjusted_iv**2 * T) / (adjusted_iv * np.sqrt(T))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        ev_value = compute_ev(adjusted_iv, rv, T, position_side)
        gamma_val = compute_gamma_value(spot, strike, adjusted_iv, T) if adjusted_iv > 0 and T > 0 else 0
        gex_value = gamma_val * ticker_data["open_interest"] * (spot**2)
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "delta": delta_est,
            "iv": adjusted_iv,
            "EV": ev_value,
            "gex": gex_value
        })
    return ticker_list

def normalize_metrics(metrics):
    arr = np.array(metrics)
    if len(arr) <= 1 or np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

def compute_composite_scores(ticker_list, position_side='short'):
    for item in ticker_list:
        if 'gex' not in item:
            item['gex'] = 0
    ev_list = [item['EV'] for item in ticker_list]
    gex_list = [item.get('gex', 0) for item in ticker_list]
    oi_list = [item['open_interest'] for item in ticker_list]
    norm_ev = normalize_metrics(ev_list)
    norm_gex = normalize_metrics(gex_list)
    norm_oi = normalize_metrics(oi_list)
    weights = {"ev": 0.5, "gex": (-0.3 if position_side.lower() == 'short' else 0.3), "oi": 0.2}
    for i, item in enumerate(ticker_list):
        composite_score = (weights["ev"] * norm_ev[i] +
                           weights["gex"] * norm_gex[i] +
                           weights["oi"] * norm_oi[i])
        item["composite_score"] = composite_score
        item["strategy"] = position_side.capitalize()
    return ticker_list

def build_smile_df(ticker_list):
    df = pd.DataFrame(ticker_list)
    df = df.dropna(subset=["iv"])
    smile_df = df.groupby("strike", as_index=False)["iv"].mean()
    return smile_df

def update_ev_for_position(ticker_list, rv, T, position_side):
    for ticker in ticker_list:
        ticker["EV"] = compute_ev(ticker["iv"], rv, T, position_side)
    return ticker_list

###########################################
# IV vs. RV COMPARISON FUNCTIONS
###########################################
# These functions compute realized volatility as provided
# and we compare that to the average IV from Thalex.

###########################################
# MAIN DASHBOARD
###########################################
def main():
    login()
    st.title("Crypto Options Visualization Dashboard (Simplified Forecast & IV vs RV)")
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

    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    global risk_factor
    risk_factor = compute_risk_adjustment_factor_cf(df_kraken, alpha=0.05)
    st.write(f"Risk Adjustment Factor (CF): {risk_factor:.2f}")
    
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
    
    # Build IV time series from Thalex mark data
    df_iv_daily = df.groupby("date_time", as_index=False)["iv_close"].mean()
    df_iv_daily["date_time"] = pd.to_datetime(df_iv_daily["date_time"])
    df_iv_daily = df_iv_daily.set_index("date_time").resample("D").mean().reset_index()
    
    # Compute realized volatility series (daily, rolling 30-day)
    rv_series = calculate_daily_realized_volatility_series(df_kraken).reset_index()
    rv_series.columns = ["date_time", "RV"]
    
    # Merge IV and RV data for comparison
    df_iv_rv = pd.merge(df_iv_daily, rv_series, on="date_time", how="outer").sort_values("date_time")
    
    st.subheader("IV vs RV Comparison")
    fig_iv_rv = go.Figure()
    fig_iv_rv.add_trace(go.Scatter(x=df_iv_rv["date_time"], y=df_iv_rv["iv_close"],
                                   mode="lines+markers", name="IV (Average)"))
    fig_iv_rv.add_trace(go.Scatter(x=df_iv_rv["date_time"], y=df_iv_rv["RV"],
                                   mode="lines+markers", name="RV (30-Day Rolling)"))
    fig_iv_rv.update_layout(title="IV vs. Realized Volatility (RV)",
                            xaxis_title="Date", yaxis_title="Volatility",
                            width=800, height=400)
    st.plotly_chart(fig_iv_rv, use_container_width=True)
    
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
    
    rv = calculate_btc_annualized_volatility_daily(df_kraken)
    global ticker_list
    ticker_list = build_ticker_list_with_metrics(all_instruments, spot_price, T_YEARS, smile_df, rv, position_side="short")
    
    df_ticker = pd.DataFrame(ticker_list)
    daily_rv_series = calculate_daily_realized_volatility_series(df_kraken)
    daily_rv = daily_rv_series.tolist()
    daily_iv = compute_daily_average_iv(df_iv_daily.set_index("date_time"))
    historical_vrps = [(iv ** 2) - (rv_val ** 2) for iv, rv_val in zip(daily_iv, daily_rv[:len(daily_iv)])]
    
    st.subheader("Volatility Trading Decision Tool")
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance",
                                          options=["Conservative", "Moderate", "Aggressive"],
                                          index=1)
    # In this simplified version, trade decision uses average IV from df and realized volatility from Kraken.
    trade_decision = {
        "iv": df["iv_close"].mean(),
        "rv": rv,
        "vol_regime": "Neutral",  # Placeholder
        "vrp_regime": "Neutral",  # Placeholder
        "put_call_ratio": 1.0,    # Placeholder
        "avg_call_delta": 0.0,    # Placeholder
        "avg_put_delta": 0.0,     # Placeholder
        "avg_call_gamma": 0.0,    # Placeholder,
        "recommendation": "Review current market conditions",
        "position": "N/A",
        "hedge_action": "N/A"
    }
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
    
    net_delta = df_ticker.assign(weighted_delta=df_ticker["delta"] * df_ticker["open_interest"])["weighted_delta"].sum()
    if net_delta > 0:
        futures_hedge = "Short BTC Futures"
    elif net_delta < 0:
        futures_hedge = "Long BTC Futures"
    else:
        futures_hedge = "No hedge required"
    hedge_table = pd.DataFrame({
        "Metric": ["Net Delta", "Futures Hedge Recommendation"],
        "Value": [net_delta, futures_hedge]
    })
    st.subheader("Futures Hedge Recommendation")
    st.dataframe(hedge_table.style.hide(axis="index"))
    
    # Additional plots (e.g., volatility smile, correlation heatmaps, etc.) can follow below.
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
    
    # Gamma Exposure plots
    df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    combined_gamma = pd.concat([df_calls, df_puts])
    plot_gamma_heatmap(combined_gamma)
    
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
        st.subheader("Gamma Exposure (GEX) by Strike")
        plot_net_gex(df_gex, spot_price)
    
if __name__ == '__main__':
    main()
