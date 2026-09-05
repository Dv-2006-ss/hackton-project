"""
Data loader for real historical freight market data.

Pulls historical daily prices for ticker BDRY (Breakwave Dry Bulk Shipping ETF),
a real-world market proxy for the Baltic Dry Index and freight futures (Capesize,
Panamax, and Supramax vessels).

Caches data locally to data/bdry_historical.csv to guarantee offline reliability
during live demos.
"""

from pathlib import Path
import pandas as pd
import warnings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BDRY_CACHE = DATA_DIR / "bdry_historical.csv"
FALLBACK_SAMPLE_DATA = DATA_DIR / "sample_freight_rates.csv"

# BDRY ETF holds dry bulk freight futures contracts.
# Its raw ETF share price (e.g. ~$12-$18) closely tracks market freight dynamics.
# We apply a scaling multiplier of 1.25 to align the ETF share price with typical
# global Capesize/Panamax bulk freight rates ($15-$25 / ton) for major trade routes
# (e.g. Australia/Indonesia to East Coast India).
PROXY_SCALING_FACTOR = 1.25


def fetch_and_cache_bdry(cache_path=DEFAULT_BDRY_CACHE, period="3y"):
    """
    Pulls historical data from Yahoo Finance via yfinance and saves to CSV.
    """
    import yfinance as yf

    print(f"Fetching live BDRY historical data (period={period}) via yfinance...")
    ticker = yf.Ticker("BDRY")
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError("yfinance returned empty dataset for ticker BDRY.")

    # Format dataframe
    df = hist[["Close"]].copy().reset_index()
    # Normalize timestamp to date (remove UTC timezone for clean daily series)
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.floor("D")
    df = df.drop_duplicates(subset=["date"]).set_index("date")

    # Regular daily calendar frequency: forward-fill non-trading days (weekends/holidays)
    # so ARIMA has contiguous daily time-steps
    df = df.asfreq("D").ffill()

    # Apply scaling transform to map ETF proxy price to realistic USD/ton freight rate
    # NOTE: This is a calibrated scaling applied to a dry bulk market ETF proxy,
    # not raw vessel charter fixture rates.
    df["freight_rate_usd_per_ton"] = (df["Close"] * PROXY_SCALING_FACTOR).round(2)

    # Clean export dataframe
    export_df = df.reset_index()[["date", "freight_rate_usd_per_ton"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(cache_path, index=False)
    print(f"Successfully cached {len(export_df)} daily records to {cache_path}")
    return export_df


def load_freight_data(cache_path=DEFAULT_BDRY_CACHE, force_refresh=False):
    """
    Loads freight rate data:
      1. If cache_path exists and not force_refresh -> load directly from CSV (offline safe).
      2. If missing or force_refresh -> fetch from yfinance, save to cache_path, and load.
      3. If yfinance fetch fails (e.g. offline) -> fallback to sample_freight_rates.csv.

    Returns:
      pandas.Series of freight_rate_usd_per_ton indexed by DatetimeIndex (daily freq).
    """
    cache_path = Path(cache_path)

    if not cache_path.exists() or force_refresh:
        try:
            fetch_and_cache_bdry(cache_path=cache_path)
        except Exception as err:
            warnings.warn(
                f"Failed to fetch live BDRY data ({err}). Falling back to cached or sample data."
            )
            if not cache_path.exists():
                if FALLBACK_SAMPLE_DATA.exists():
                    warnings.warn(f"Using fallback synthetic dataset: {FALLBACK_SAMPLE_DATA}")
                    cache_path = FALLBACK_SAMPLE_DATA
                else:
                    raise FileNotFoundError(
                        f"Neither {cache_path} nor fallback {FALLBACK_SAMPLE_DATA} exists."
                    )

    df = pd.read_csv(cache_path, parse_dates=["date"])
    df = df.set_index("date")
    # Ensure regular daily frequency
    if df.index.freq is None:
        df = df.asfreq("D").ffill()

    return df["freight_rate_usd_per_ton"]


if __name__ == "__main__":
    series = load_freight_data(force_refresh=True)
    print("\nData summary:")
    print(f"Total observations: {len(series)}")
    print(f"Date range: {series.index.min().date()} to {series.index.max().date()}")
    print(f"Rate range: ${series.min():.2f} to ${series.max():.2f} / ton")
    print(f"Mean rate: ${series.mean():.2f} / ton")
    print("\nFirst 5 days:")
    print(series.head(5))
    print("\nLatest 5 days:")
    print(series.tail(5))
