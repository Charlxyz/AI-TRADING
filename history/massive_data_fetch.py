"""
fetch_btcusd.py
---------------
Télécharge l'historique complet BTCUSDT (Binance) sur le timeframe choisi.

Format de sortie :
    datetime,open,high,low,close,volume

Timeframes supportés : 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

Usage :
    pip install requests pandas tqdm
    python fetch_btcusd.py              # → daily par défaut
    python fetch_btcusd.py 15m
    python fetch_btcusd.py 5m
    python fetch_btcusd.py 1h
"""

import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─── Configuration ────────────────────────────────────────────────────────────

SYMBOL        = "BTCUSDT"
BASE_URL      = "https://api.binance.com/api/v3/klines"
LIMIT         = 1000          # max bougies par requête (limite Binance)
START_DATE    = "2017-08-17"  # premier jour disponible sur Binance
PAUSE_SEC     = 0.15          # pause entre requêtes (évite le rate-limit)


# ─── Estimation du nombre total de bougies ────────────────────────────────────

MINUTES_PER_INTERVAL = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
}

def estimate_total_candles(interval: str) -> int:
    mins_per_candle = MINUTES_PER_INTERVAL.get(interval, 1)
    start = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now   = datetime.now(timezone.utc)
    total_minutes = (now - start).total_seconds() / 60
    return int(total_minutes / mins_per_candle)


# ─── Fetch Binance ────────────────────────────────────────────────────────────

def fetch_binance(interval: str) -> pd.DataFrame:
    start_ms = int(
        datetime.strptime(START_DATE, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    estimated = estimate_total_candles(interval)
    total_requests = (estimated // LIMIT) + 1

    print(f"\n📡 Binance — {SYMBOL} [{interval}]")
    print(f"   Période  : {START_DATE} → aujourd'hui")
    print(f"   ~{estimated:,} bougies  |  ~{total_requests} requêtes\n")

    if HAS_TQDM:
        pbar = tqdm(total=total_requests, unit="req", ncols=70)

    all_rows = []
    req_count = 0

    while start_ms < now_ms:
        params = {
            "symbol":    SYMBOL,
            "interval":  interval,
            "startTime": start_ms,
            "limit":     LIMIT,
        }

        for attempt in range(5):
            try:
                r = requests.get(BASE_URL, params=params, timeout=15)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"\n⚠️  Erreur ({e}), retry dans {wait}s...")
                time.sleep(wait)
        else:
            print("❌ Échec après 5 tentatives.")
            break

        data = r.json()
        if not data:
            break

        all_rows.extend(data)
        start_ms = data[-1][0] + 1  # reprend juste après la dernière bougie
        req_count += 1

        if HAS_TQDM:
            pbar.update(1)
            pbar.set_postfix({"bougies": f"{len(all_rows):,}"})
        elif req_count % 50 == 0:
            print(f"   {req_count} requêtes | {len(all_rows):,} bougies...")

        time.sleep(PAUSE_SEC)

    if HAS_TQDM:
        pbar.close()

    # ── Mise en forme ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Format datetime selon le timeframe
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    if interval in ("1d", "3d", "1w", "1M"):
        df["datetime"] = ts.dt.strftime("%Y-%m-%d")
    else:
        df["datetime"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df[["datetime", "open", "high", "low", "close", "volume"]].astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    interval = sys.argv[1] if len(sys.argv) > 1 else "1d"

    if interval not in MINUTES_PER_INTERVAL:
        print(f"❌ Timeframe inconnu : '{interval}'")
        print(f"   Valides : {', '.join(MINUTES_PER_INTERVAL.keys())}")
        sys.exit(1)

    output_file = f"BTCUSD_{interval}_history.csv"

    df = fetch_binance(interval)

    if df is None or df.empty:
        print("❌ Aucune donnée récupérée.")
        sys.exit(1)

    # Nettoyage
    df = (
        df.drop_duplicates(subset="datetime")
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    df.to_csv(output_file, index=False, float_format="%.8f")

    size_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"\n✅ Fichier généré : {output_file}")
    print(f"   Période        : {df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]}")
    print(f"   Lignes totales : {len(df):,}")
    print(f"   Taille ~       : {size_mb:.1f} MB en mémoire")
    print(f"\nAperçu :")
    print(df.head(3).to_string(index=False))
    print("...")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()