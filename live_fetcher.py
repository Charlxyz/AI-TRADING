"""
live_fetcher.py — Récupération des données Binance en temps réel
Utilise l'API publique Binance (pas de clé API nécessaire pour les données de marché).

Fournit :
  - Les bougies HTF (15m) et LTF (5m) pour le modèle
  - La mise à jour en continu bougie par bougie
"""

import time
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional


BINANCE_BASE = "https://api.binance.com/api/v3"

# Mapping intervalle → durée en secondes
INTERVAL_SECONDS = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1h":  3600,
}


def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Télécharge les dernières `limit` bougies depuis Binance.
    Retourne un DataFrame avec colonnes [open, high, low, close, volume].
    """
    url = f"{BINANCE_BASE}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    rows = []
    for r in raw:
        rows.append({
            "datetime": pd.to_datetime(r[0], unit="ms", utc=True),
            "open":   float(r[1]),
            "high":   float(r[2]),
            "low":    float(r[3]),
            "close":  float(r[4]),
            "volume": float(r[5]),
        })

    df = pd.DataFrame(rows).set_index("datetime")

    # On exclut la bougie en cours (pas encore fermée) sauf si demandé
    # La dernière bougie Binance est toujours la courante → on la retire
    # pour n'avoir que des bougies clôturées.
    if len(df) > 1:
        df = df.iloc[:-1]

    return df


def get_current_price(symbol: str) -> float:
    """Prix actuel via le ticker Binance."""
    url = f"{BINANCE_BASE}/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])


def seconds_to_next_close(interval: str) -> float:
    """
    Calcule le nombre de secondes avant la fermeture de la bougie courante.
    Utile pour savoir quand rafraîchir les données.
    """
    dur = INTERVAL_SECONDS.get(interval, 60)
    now_ts = time.time()
    elapsed = now_ts % dur
    return dur - elapsed


class LiveDataFeed:
    """
    Flux de données live pour HTF (15m) et LTF (5m).
    Maintient un buffer glissant de bougies clôturées.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        htf: str = "15m",
        ltf: str = "5m",
        htf_lookback: int = 100,   # bougies HTF à conserver
        ltf_lookback: int = 300,   # bougies LTF à conserver
    ):
        self.symbol = symbol
        self.htf_interval = htf
        self.ltf_interval = ltf
        self.htf_lookback = htf_lookback
        self.ltf_lookback = ltf_lookback

        self.htf_candles: Optional[pd.DataFrame] = None
        self.ltf_candles: Optional[pd.DataFrame] = None
        self.last_htf_time: Optional[pd.Timestamp] = None
        self.last_ltf_time: Optional[pd.Timestamp] = None
        self.current_price: float = 0.0
        self.last_refresh: float = 0.0
        self.errors: list[str] = []

    def initialize(self) -> bool:
        """Charge les données initiales. Retourne True si succès."""
        try:
            self.htf_candles = fetch_klines(self.symbol, self.htf_interval, self.htf_lookback + 1)
            self.ltf_candles = fetch_klines(self.symbol, self.ltf_interval, self.ltf_lookback + 1)
            self.current_price = get_current_price(self.symbol)
            self.last_htf_time = self.htf_candles.index[-1]
            self.last_ltf_time = self.ltf_candles.index[-1]
            self.last_refresh = time.time()
            return True
        except Exception as e:
            self.errors.append(f"Init error: {e}")
            return False

    def refresh(self) -> dict:
        """
        Rafraîchit les données.
        Retourne un dict avec :
          - new_htf_candle : True si une nouvelle bougie HTF s'est fermée
          - new_ltf_candle : True si une nouvelle bougie LTF s'est fermée
          - current_price  : prix actuel
        """
        result = {"new_htf_candle": False, "new_ltf_candle": False, "current_price": 0.0}

        try:
            self.current_price = get_current_price(self.symbol)
            result["current_price"] = self.current_price

            # Refresh LTF (toutes les ~30s pour détecter rapidement les nouvelles bougies)
            new_ltf = fetch_klines(self.symbol, self.ltf_interval, 10)
            if len(new_ltf) > 0 and new_ltf.index[-1] != self.last_ltf_time:
                # Nouvelle(s) bougie(s) LTF
                self.ltf_candles = pd.concat([self.ltf_candles, new_ltf[new_ltf.index > self.last_ltf_time]])
                self.ltf_candles = self.ltf_candles[~self.ltf_candles.index.duplicated(keep="last")]
                self.ltf_candles = self.ltf_candles.tail(self.ltf_lookback)
                self.last_ltf_time = self.ltf_candles.index[-1]
                result["new_ltf_candle"] = True

            # Refresh HTF
            new_htf = fetch_klines(self.symbol, self.htf_interval, 5)
            if len(new_htf) > 0 and new_htf.index[-1] != self.last_htf_time:
                self.htf_candles = pd.concat([self.htf_candles, new_htf[new_htf.index > self.last_htf_time]])
                self.htf_candles = self.htf_candles[~self.htf_candles.index.duplicated(keep="last")]
                self.htf_candles = self.htf_candles.tail(self.htf_lookback)
                self.last_htf_time = self.htf_candles.index[-1]
                result["new_htf_candle"] = True

            self.last_refresh = time.time()

        except Exception as e:
            self.errors.append(f"{datetime.now().strftime('%H:%M:%S')} — {e}")
            if len(self.errors) > 20:
                self.errors = self.errors[-20:]

        return result

    def time_to_next_htf(self) -> float:
        """Secondes avant la prochaine bougie HTF."""
        return seconds_to_next_close(self.htf_interval)

    def time_to_next_ltf(self) -> float:
        """Secondes avant la prochaine bougie LTF."""
        return seconds_to_next_close(self.ltf_interval)

    @property
    def is_ready(self) -> bool:
        return (
            self.htf_candles is not None
            and self.ltf_candles is not None
            and len(self.htf_candles) >= 20
            and len(self.ltf_candles) >= 10
        )


# ──────────────────────────────────────────────────────────────────────────────
# MarketDataHub — fetch mutualisé pour le multi-actifs (server.py)
# ──────────────────────────────────────────────────────────────────────────────

class MarketDataHub:
    """
    Point d'accès unique aux données Binance pour tous les workers du serveur
    multi-actifs.

    Problème résolu : dans config.json, plusieurs "workers" utilisent souvent
    le même (symbole, intervalle) — ex. BTCUSDT en 15m est utilisé à la fois
    par le worker "15m/1m" et le worker "15m/5m". Avec un LiveDataFeed par
    worker, ce même (symbole, intervalle) est téléchargé deux fois par cycle
    pour rien.

    Le hub télécharge chaque (symbole, intervalle) UNE SEULE fois par cycle,
    et chaque worker qui en a besoin lit une copie de ce cache au lieu
    d'appeler l'API lui-même. Le prix courant (ticker) est mutualisé de la
    même façon, une fois par symbole.

    Thread-safety : toutes les lectures/écritures passent par un verrou
    (`self._lock`), donc plusieurs workers peuvent lire pendant qu'un seul
    thread (le MarketPoller côté server.py) écrit.
    """

    def __init__(self):
        self._candles: dict[tuple[str, str], pd.DataFrame] = {}
        self._last_time: dict[tuple[str, str], "pd.Timestamp"] = {}
        self._lookback: dict[tuple[str, str], int] = {}
        self._prices: dict[str, float] = {}
        self._lock = threading.Lock()
        self.errors: list[str] = []

    def register(self, symbol: str, interval: str, lookback: int):
        """
        Déclare qu'un worker a besoin de ce (symbole, intervalle), avec un
        lookback minimum. Idempotent : appeler plusieurs fois avec la même
        paire ne provoque pas de fetch supplémentaire, seul le lookback max
        demandé est retenu.
        """
        key = (symbol, interval)
        with self._lock:
            self._lookback[key] = max(self._lookback.get(key, 0), lookback)

    def initialize(self) -> bool:
        """
        Charge les données initiales pour tous les (symbole, intervalle)
        enregistrés via `register()`. À appeler une fois, après avoir
        enregistré tous les workers, avant de démarrer le rafraîchissement.
        """
        with self._lock:
            pending = list(self._lookback.items())

        ok = True
        symbols_done = set()
        for (symbol, interval), lookback in pending:
            try:
                df = fetch_klines(symbol, interval, lookback + 1)
                with self._lock:
                    self._candles[(symbol, interval)] = df
                    self._last_time[(symbol, interval)] = df.index[-1]
            except Exception as e:
                self.errors.append(f"Init {symbol} {interval} : {e}")
                ok = False

            if symbol not in symbols_done:
                symbols_done.add(symbol)
                try:
                    price = get_current_price(symbol)
                    with self._lock:
                        self._prices[symbol] = price
                except Exception as e:
                    self.errors.append(f"Init prix {symbol} : {e}")
                    ok = False

        return ok

    def refresh(self):
        """
        Un seul passage de rafraîchissement pour TOUT le serveur :
        - une requête ticker par symbole unique (pas par worker)
        - une requête klines par (symbole, intervalle) unique (pas par worker)
        À appeler périodiquement par un unique thread (le MarketPoller).
        """
        with self._lock:
            keys = list(self._lookback.keys())
            symbols = sorted({s for s, _ in keys})

        for symbol in symbols:
            try:
                price = get_current_price(symbol)
                with self._lock:
                    self._prices[symbol] = price
            except Exception as e:
                self._log_error(f"Prix {symbol} : {e}")

        for (symbol, interval) in keys:
            try:
                new_data = fetch_klines(symbol, interval, 10)
                if len(new_data) == 0:
                    continue
                with self._lock:
                    last_time = self._last_time.get((symbol, interval))
                    if last_time is not None and new_data.index[-1] == last_time:
                        continue  # rien de neuf pour cette paire

                    old = self._candles.get((symbol, interval))
                    if old is not None and last_time is not None:
                        merged = pd.concat([old, new_data[new_data.index > last_time]])
                    else:
                        merged = new_data
                    merged = merged[~merged.index.duplicated(keep="last")]

                    lookback = self._lookback.get((symbol, interval), 200)
                    merged = merged.tail(lookback)
                    self._candles[(symbol, interval)] = merged
                    self._last_time[(symbol, interval)] = merged.index[-1]
            except Exception as e:
                self._log_error(f"{symbol} {interval} : {e}")

    def _log_error(self, msg: str):
        self.errors.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")
        if len(self.errors) > 30:
            self.errors = self.errors[-30:]

    # ── Lecture (utilisée par les workers) ───────────────────────────

    def get_candles(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        with self._lock:
            df = self._candles.get((symbol, interval))
            return df.copy() if df is not None else None

    def get_last_time(self, symbol: str, interval: str):
        with self._lock:
            return self._last_time.get((symbol, interval))

    def get_price(self, symbol: str) -> float:
        with self._lock:
            return self._prices.get(symbol, 0.0)

    def candle_count(self, symbol: str, interval: str) -> int:
        with self._lock:
            df = self._candles.get((symbol, interval))
            return len(df) if df is not None else 0

    def time_to_next_close(self, interval: str) -> float:
        return seconds_to_next_close(interval)