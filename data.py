"""Download and cache S&P 500 (^GSPC) daily OHLC data from Yahoo Finance."""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "spx.csv"


def download_spx(range_: str = "15y", cache: bool = True) -> pd.DataFrame:
    """Return daily OHLC frame indexed by date. Uses the cached CSV when present."""
    if cache and DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
        return df
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range={range_}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(urllib.request.urlopen(req, timeout=30).read())
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"]},
        index=pd.to_datetime(ts, unit="s").normalize(),
    )
    df.index.name = "Date"
    df = df.dropna()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH)
    # be polite to the free endpoint
    time.sleep(1)
    return df


if __name__ == "__main__":
    df = download_spx(cache=False)
    print(df.tail())
    print(f"{len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
