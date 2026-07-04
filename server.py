"""
server.py — Serveur multi-actifs FVG Base Hits
Gère N actifs en parallèle (threads) et diffuse les données
en temps réel via WebSocket vers le dashboard web.

Usage :
    python server.py --model models/BEST_MODEL/best_model.zip
    python server.py --model models/BEST_MODEL/best_model.zip --config config.json
    python server.py --model models/BEST_MODEL/best_model.zip --port 8765
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Imports projet ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_fetcher import MarketDataHub
from paper_engine import PaperTradingEngine

# ── WebSocket ────────────────────────────────────────────────────────
try:
    import websockets
    import websockets.server
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("⚠  websockets non installé — lance : pip install websockets")

logging.basicConfig(level=logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Instance par actif
# ──────────────────────────────────────────────────────────────────────────────

class SymbolWorker:
    """
    Gère un actif : moteur paper trading uniquement.

    Contrairement à la version précédente, ce worker ne fait PLUS aucun appel
    réseau lui-même et ne tourne plus dans son propre thread. Les données de
    marché (bougies HTF/LTF + prix) viennent d'un MarketDataHub partagé entre
    tous les workers : si deux workers utilisent le même (symbole, intervalle)
    — ex. BTCUSDT en 15m pour les combos 15m/1m ET 15m/5m — la donnée n'est
    téléchargée qu'une seule fois et lue par les deux.

    Le worker s'enregistre auprès du hub à la création (`hub.register(...)`),
    puis sa méthode `tick()` est appelée périodiquement par le MarketPoller
    central : il lit ce que le hub a déjà en cache, détecte si SA bougie HTF
    a changé, et déclenche l'agent si c'est le cas.
    """

    def __init__(self, cfg: dict, model_path: str, global_cfg: dict, hub: MarketDataHub):
        self.symbol      = cfg["symbol"]
        self.htf         = cfg.get("htf", "15m")
        self.ltf         = cfg.get("ltf", "1m")
        self.balance_init = cfg.get("balance", global_cfg.get("initial_balance", 1000.0))
        self.risk_pct    = cfg.get("risk_pct", global_cfg.get("risk_pct", 0.01))
        self.leverage    = cfg.get("leverage", global_cfg.get("leverage", 1))
        self.model_path  = model_path
        self.hub         = hub

        # Nom unique : BTCUSDT_15m_1m → deux configs du même actif ne s'écrasent pas
        self.run_id   = f"{self.symbol}_{self.htf}_{self.ltf}"
        session_dir   = os.path.join("paper_sessions", self.symbol)
        os.makedirs(session_dir, exist_ok=True)
        # run_id = BTCUSDT_15m_1m → fichier BTCUSDT_15m_1m.json (jamais de double extension)
        run_id_clean  = self.run_id.replace(".json", "")
        self.log_path = os.path.join(session_dir, f"{run_id_clean}.json")

        # État partagé (lu par le serveur WebSocket)
        self.lock   = threading.Lock()
        self.status = "init"          # init | ok | error
        self.error  = ""
        self.last_result: dict = {}
        self.last_update = ""

        self.engine: Optional[PaperTradingEngine] = None
        self._last_htf_seen = None    # dernier timestamp HTF déjà traité par CE worker

        # Déclare ses besoins auprès du hub. Idempotent : si un autre worker a
        # déjà enregistré le même (symbole, intervalle), aucun fetch en double
        # n'est créé — seul le lookback max demandé est retenu.
        self.hub.register(self.symbol, self.htf, lookback=150)
        self.hub.register(self.symbol, self.ltf, lookback=900)

    # ── Démarrage ───────────────────────────────────────────────────

    def load_model(self) -> bool:
        """Charge le moteur PPO. Retourne True si succès. Pas d'appel réseau ici."""
        try:
            self.engine = PaperTradingEngine(
                model_path=self.model_path,
                initial_balance=self.balance_init,
                risk_pct=self.risk_pct,
                leverage=self.leverage,
                log_path=self.log_path,
            )
            if not self.engine.model_loaded:
                self._set_error(f"Modèle non chargé : {self.engine.model_error}")
                return False
        except Exception as e:
            self._set_error(str(e))
            return False

        with self.lock:
            self.status = "ok"
            self.last_update = datetime.now().strftime("%H:%M:%S")
        return True

    def _set_error(self, msg: str):
        with self.lock:
            self.status = "error"
            self.error  = msg
        print(f"[{self.symbol}] ERREUR : {msg}")

    # ── Cycle appelé par le MarketPoller central ─────────────────────

    def tick(self):
        """
        Appelé à chaque cycle, une fois que le hub a rafraîchi ses données.
        Ne fait aucune requête Binance : lit uniquement ce que le hub a déjà
        en cache, propre à ce worker (sa balance, ses trades, sa position
        restent individuels — seule la donnée de marché brute est partagée).
        """
        if self.status != "ok" or self.engine is None:
            return

        try:
            price = self.hub.get_price(self.symbol)
            htf_candles = self.hub.get_candles(self.symbol, self.htf)
            ltf_candles = self.hub.get_candles(self.symbol, self.ltf)

            if htf_candles is None or ltf_candles is None:
                return

            ready = len(htf_candles) >= 20 and len(ltf_candles) >= 10
            htf_time = self.hub.get_last_time(self.symbol, self.htf)
            new_htf_candle = ready and htf_time is not None and htf_time != self._last_htf_seen

            if new_htf_candle:
                self._last_htf_seen = htf_time
                engine_result = self.engine.on_new_candle(htf_candles, ltf_candles, price)
                with self.lock:
                    self.last_result = engine_result
                    self.last_update = datetime.now().strftime("%H:%M:%S")

            # PnL latent de la position ouverte, mis à jour à chaque tick
            # (indépendamment d'une nouvelle bougie HTF ou non)
            if self.engine.open_trade and price > 0:
                t = self.engine.open_trade
                if t.direction == "buy":
                    t.pnl_usdt = (price - t.entry_price) * t.position_size
                else:
                    t.pnl_usdt = (t.entry_price - price) * t.position_size

            with self.lock:
                self.last_update = datetime.now().strftime("%H:%M:%S")

        except Exception as e:
            with self.lock:
                self.error = str(e)

    # ── Snapshot JSON pour WebSocket ─────────────────────────────────

    def snapshot(self) -> dict:
        """Retourne un dict JSON-sérialisable de l'état courant."""
        with self.lock:
            engine = self.engine

        if engine is None:
            return {
                "symbol": self.symbol, "htf": self.htf, "ltf": self.ltf,
                "status": self.status, "error": self.error,
                "price": 0, "balance": self.balance_init,
                "pnl": 0, "pnl_pct": 0,
                "open_trade": None, "trades": [],
                "stats": {}, "equity_curve": [],
                "last_result": {}, "last_update": self.last_update,
                "time_to_htf": 0, "htf_count": 0, "ltf_count": 0,
            }

        stats = engine.get_stats()
        pnl   = engine.balance - engine.initial_balance

        open_trade = None
        if engine.open_trade:
            t = engine.open_trade
            open_trade = {
                "direction":   t.direction,
                "entry_price": t.entry_price,
                "stop_loss":   t.stop_loss,
                "take_profit": t.take_profit,
                "pnl_usdt":    round(t.pnl_usdt, 2),
                "had_fvg":     t.had_fvg,
                "had_ltf":     t.had_ltf_model,
                "leverage":    t.leverage,
                "entry_time":  t.entry_time,
            }

        trades_out = []
        for t in engine.trades:
            trades_out.append({
                "id":          t.id,
                "direction":   t.direction,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "stop_loss":   t.stop_loss,
                "take_profit": t.take_profit,
                "pnl_usdt":    round(t.pnl_usdt, 2),
                "pnl_pct":     round(t.pnl_pct, 2),
                "status":      t.status,
                "had_fvg":     t.had_fvg,
                "had_ltf":     t.had_ltf_model,
                "leverage":    getattr(t, "leverage", 1),
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
            })

        lr = self.last_result
        return {
            "symbol":      self.symbol,
            "run_id":      self.run_id,
            "htf":         self.htf,
            "ltf":         self.ltf,
            "status":      self.status,
            "error":       self.error,
            "price":       self.hub.get_price(self.symbol),
            "balance":     round(engine.balance, 2),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl / engine.initial_balance * 100, 2),
            "open_trade":  open_trade,
            "trades":      list(reversed(trades_out)),
            "stats":       {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in stats.items()},
            "equity_curve": engine.equity_curve[-200:],
            "last_result": {
                "action_label": lr.get("action_label", "—"),
                "fvg_active":   lr.get("fvg_active", False),
                "ltf_signal":   lr.get("ltf_signal", False),
                "conformity":   lr.get("conformity", False),
                "message":      lr.get("message", ""),
            },
            "last_update":  self.last_update,
            "time_to_htf":  round(self.hub.time_to_next_close(self.htf)),
            "htf_count":    self.hub.candle_count(self.symbol, self.htf),
            "ltf_count":    self.hub.candle_count(self.symbol, self.ltf),
            "leverage":     self.leverage,
        }


# ──────────────────────────────────────────────────────────────────────────────
# MarketPoller — thread unique qui pilote tout le rafraîchissement
# ──────────────────────────────────────────────────────────────────────────────

class MarketPoller(threading.Thread):
    """
    Remplace les N threads individuels (un par worker) qui existaient avant.
    Un seul thread :
      1. demande au hub de se rafraîchir (une requête par (symbole, intervalle)
         unique et une requête ticker par symbole unique, quel que soit le
         nombre de workers qui les utilisent)
      2. appelle ensuite `tick()` sur chaque worker, qui lit le cache du hub
         et ne fait donc aucun appel réseau supplémentaire.
    """

    def __init__(self, hub: MarketDataHub, workers: list[SymbolWorker], interval: int = 15):
        super().__init__(daemon=True, name="market-poller")
        self.hub = hub
        self.workers = workers
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            try:
                self.hub.refresh()
            except Exception as e:
                print(f"[MarketPoller] Erreur de rafraîchissement : {e}")
            for w in self.workers:
                w.tick()
            time.sleep(self.interval)

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────────────────────────────────────
# Serveur WebSocket
# ──────────────────────────────────────────────────────────────────────────────

class MultiAssetServer:
    """
    Serveur WebSocket qui :
    - Démarre un SymbolWorker par actif activé
    - Diffuse les snapshots toutes les secondes à tous les clients connectés
    - Accepte les commandes (close_position, toggle_symbol)
    """

    def __init__(self, workers: list[SymbolWorker], port: int = 8765):
        self.workers  = {w.run_id: w for w in workers}
        self.port     = port
        self.clients  = set()
        self.lock     = asyncio.Lock()

    async def register(self, ws):
        async with self.lock:
            self.clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            async with self.lock:
                self.clients.discard(ws)

    async def broadcast(self):
        """Envoie un snapshot complet à tous les clients toutes les secondes."""
        while True:
            await asyncio.sleep(1)
            if not self.clients:
                continue
            payload = json.dumps({
                "type": "update",
                "ts":   datetime.now().strftime("%H:%M:%S"),
                "symbols": {sym: w.snapshot() for sym, w in self.workers.items()},
            })
            dead = set()
            async with self.lock:
                clients = set(self.clients)
            for ws in clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            async with self.lock:
                self.clients -= dead

    async def handle(self, ws):
        """Gère les messages entrants d'un client."""
        await self.register(ws)

    async def handle_commands(self, ws):
        """Version avec lecture des commandes."""
        async with self.lock:
            self.clients.add(ws)
        try:
            async for msg in ws:
                try:
                    cmd = json.loads(msg)
                    await self._process_command(cmd, ws)
                except Exception as e:
                    await ws.send(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass
        finally:
            async with self.lock:
                self.clients.discard(ws)

    async def _process_command(self, cmd: dict, ws):
        action = cmd.get("action")
        symbol = cmd.get("symbol")

        if action == "close_position" and symbol in self.workers:  # symbol = run_id ici
            w = self.workers[symbol]
            if w.engine and w.engine.open_trade:
                w.engine.force_close(w.hub.get_price(w.symbol))
                await ws.send(json.dumps({
                    "type": "info",
                    "msg": f"{symbol} : position clôturée manuellement"
                }))

        elif action == "ping":
            await ws.send(json.dumps({"type": "pong"}))

    async def serve(self):
        print(f"\n🌐 Dashboard disponible sur : http://localhost:{self.port}/")
        print(f"   WebSocket              sur : ws://localhost:{self.port+1}/")
        print("   (ouvre dashboard.html dans ton navigateur)\n")
        async with websockets.serve(self.handle_commands, "0.0.0.0", self.port + 1):
            await self.broadcast()


# ──────────────────────────────────────────────────────────────────────────────
# Serveur HTTP minimal pour servir dashboard.html
# ──────────────────────────────────────────────────────────────────────────────

def start_http_server(port: int):
    """Sert les fichiers statiques (dashboard.html) sur le port HTTP."""
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass  # silencieux

    def run():
        with socketserver.TCPServer(("", port), Handler) as httpd:
            httpd.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FVG Multi-Asset Paper Trading Server")
    parser.add_argument("--model",  type=str, required=True, help="Chemin vers le modèle .zip")
    parser.add_argument("--config", type=str, default="config.json", help="Fichier de config")
    parser.add_argument("--port",   type=int, default=8765, help="Port HTTP (WS = port+1)")
    args = parser.parse_args()

    if not WS_AVAILABLE:
        print("❌ Installe websockets : pip install websockets")
        sys.exit(1)

    # Lecture config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ Config introuvable : {args.config}")
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    global_cfg = cfg.get("global", {})
    port = args.port or global_cfg.get("web_port", 8765)

    symbols_cfg = [s for s in cfg.get("symbols", []) if s.get("enabled", True)]
    if not symbols_cfg:
        print("❌ Aucun symbole activé dans config.json")
        sys.exit(1)

    print("\n" + "═" * 55)
    print("  FVG BASE HITS — SERVEUR MULTI-ACTIFS")
    print("═" * 55)
    print(f"  Modèle  : {args.model}")
    print(f"  Config  : {args.config}")
    print(f"  Actifs  : {', '.join(s['symbol'] for s in symbols_cfg)}")
    print(f"  Port    : {port} (HTTP) / {port+1} (WebSocket)")
    print()

    refresh_interval = global_cfg.get("refresh_interval", 15)

    # 1. Un seul hub de données pour tout le serveur : chaque worker s'y
    #    enregistre, mais les (symbole, intervalle) identiques entre workers
    #    ne sont téléchargés qu'une fois.
    hub = MarketDataHub()
    workers = [SymbolWorker(s_cfg, args.model, global_cfg, hub) for s_cfg in symbols_cfg]

    n_pairs = len({(s_cfg["symbol"], s_cfg["htf"]) for s_cfg in symbols_cfg} |
                  {(s_cfg["symbol"], s_cfg["ltf"]) for s_cfg in symbols_cfg})
    n_symbols = len({s_cfg["symbol"] for s_cfg in symbols_cfg})
    print(f"  ⏳ {len(workers)} workers → {n_pairs} paires (symbole, intervalle) uniques "
          f"+ {n_symbols} prix uniques à récupérer par cycle (au lieu de "
          f"{len(workers) * 3} requêtes sans mutualisation)")

    if not hub.initialize():
        print("⚠️  Certaines données initiales n'ont pas pu être chargées :")
        for err in hub.errors:
            print(f"   {err}")

    # 2. Chargement des modèles PPO (pas d'appel réseau ici)
    for w in workers:
        print(f"  ⏳ Chargement du modèle pour {w.symbol} ({w.htf}/{w.ltf})...")
        w.load_model()

    # 3. Un seul thread pilote le rafraîchissement de TOUS les workers
    poller = MarketPoller(hub, workers, interval=refresh_interval)
    poller.start()

    # Serveur HTTP pour dashboard.html
    start_http_server(port)

    # Serveur WebSocket
    server = MultiAssetServer(workers, port)

    print("\n✅ Serveur démarré — Ctrl+C pour arrêter")
    print(f"   👉 Ouvre http://localhost:{port}/dashboard.html\n")

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\n⏹  Arrêt — sauvegarde des sessions...")
        poller.stop()
        for w in workers:
            if w.engine:
                w.engine._save_history()
        print("✅ Sessions sauvegardées.")


if __name__ == "__main__":
    main()