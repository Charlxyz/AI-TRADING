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

# ── Imports projet ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_fetcher import LiveDataFeed
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
    Gère un actif : fetcher Binance + moteur paper trading.
    Tourne dans son propre thread.
    """

    def __init__(self, cfg: dict, model_path: str, global_cfg: dict):
        self.symbol      = cfg["symbol"]
        self.htf         = cfg.get("htf", "15m")
        self.ltf         = cfg.get("ltf", "1m")
        self.balance_init = cfg.get("balance", global_cfg.get("initial_balance", 1000.0))
        self.risk_pct    = cfg.get("risk_pct", global_cfg.get("risk_pct", 0.01))
        self.leverage    = cfg.get("leverage", global_cfg.get("leverage", 1))
        self.refresh_sec = global_cfg.get("refresh_interval", 15)
        self.model_path  = model_path

        # Filtre de session — peut être défini globalement ou par symbole
        # Format : liste de [heure_debut, heure_fin] en UTC
        # Ex: [[7, 23]] = bloquer la session asiatique (23h-7h UTC)
        raw = cfg.get("session_filter", global_cfg.get("session_filter", None))
        self.session_filter = [tuple(r) for r in raw] if raw else []
        self.session_tz_offset = cfg.get("tz_offset", global_cfg.get("tz_offset", 0))

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

        self.feed   = None
        self.engine = None
        self._thread: threading.Thread = None
        self._running = False

    # ── Démarrage ───────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"worker-{self.symbol}")
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        """Boucle principale du worker."""
        try:
            # 1. Moteur
            self.engine = PaperTradingEngine(
                model_path=self.model_path,
                initial_balance=self.balance_init,
                risk_pct=self.risk_pct,
                leverage=self.leverage,
                log_path=self.log_path,
            )
            if not self.engine.model_loaded:
                self._set_error(f"Modèle non chargé : {self.engine.model_error}")
                return

            # Appliquer le filtre de session si configuré
            if self.session_filter:
                self.engine.set_session_filter(self.session_filter, self.session_tz_offset)
                ranges_str = ", ".join(f"{s}h-{e}h UTC" for s,e in self.session_filter)
                print(f"  [{self.symbol}] Filtre session actif : {ranges_str}")

            # 2. Feed
            self.feed = LiveDataFeed(
                symbol=self.symbol,
                htf=self.htf,
                ltf=self.ltf,
                htf_lookback=150,
                ltf_lookback=900,
            )
            if not self.feed.initialize():
                self._set_error(f"Binance init failed : {self.feed.errors}")
                return

            with self.lock:
                self.status = "ok"
                self.last_update = datetime.now().strftime("%H:%M:%S")

            # 3. Boucle principale
            while self._running:
                try:
                    result = self.feed.refresh()
                    price  = result.get("current_price", 0.0)

                    if result.get("new_htf_candle") and self.feed.is_ready:
                        engine_result = self.engine.on_new_candle(
                            self.feed.htf_candles,
                            self.feed.ltf_candles,
                            price,
                        )
                        with self.lock:
                            self.last_result  = engine_result
                            self.last_update  = datetime.now().strftime("%H:%M:%S")

                    # PnL latent de la position ouverte
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

                time.sleep(self.refresh_sec)

        except Exception as e:
            self._set_error(str(e))

    def _set_error(self, msg: str):
        with self.lock:
            self.status = "error"
            self.error  = msg
        print(f"[{self.symbol}] ERREUR : {msg}")

    # ── Snapshot JSON pour WebSocket ─────────────────────────────────

    def snapshot(self) -> dict:
        """Retourne un dict JSON-sérialisable de l'état courant."""
        with self.lock:
            engine = self.engine
            feed   = self.feed

        if engine is None or feed is None:
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
            "price":       feed.current_price,
            "balance":     round(engine.balance, 2),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl / engine.initial_balance * 100, 2),
            "open_trade":  open_trade,
            "trades":      list(reversed(trades_out)),
            "stats":       {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in stats.items()},
            "equity_curve": engine.equity_curve,  # courbe complète pour le modal
            "last_result": {
                "action_label": lr.get("action_label", "—"),
                "fvg_active":   lr.get("fvg_active", False),
                "ltf_signal":   lr.get("ltf_signal", False),
                "conformity":   lr.get("conformity", False),
                "message":      lr.get("message", ""),
            },
            "last_update":  self.last_update,
            "time_to_htf":  round(feed.time_to_next_htf()),
            "htf_count":    len(feed.htf_candles) if feed.htf_candles is not None else 0,
            "ltf_count":    len(feed.ltf_candles) if feed.ltf_candles is not None else 0,
            "leverage":     self.leverage,
        }


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
            if w.engine and w.engine.open_trade and w.feed:
                w.engine.force_close(w.feed.current_price)
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

    # Démarrage des workers
    workers = []
    for s_cfg in symbols_cfg:
        print(f"  ⏳ Démarrage {s_cfg['symbol']} ({s_cfg['htf']}/{s_cfg['ltf']})...")
        w = SymbolWorker(s_cfg, args.model, global_cfg)
        w.start()
        workers.append(w)
        time.sleep(0.5)  # évite de saturer l'API Binance

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
        for w in workers:
            w.stop()
            if w.engine:
                w.engine._save_history()
        print("✅ Sessions sauvegardées.")


if __name__ == "__main__":
    main()