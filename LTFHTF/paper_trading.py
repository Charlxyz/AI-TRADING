"""
paper_trading.py — Script principal du paper trading FVG Base Hits

Usage :
    python paper_trading.py --model models/best_model.zip
    python paper_trading.py --model models/best_model.zip --balance 5000 --risk 0.005
    python paper_trading.py --model models/best_model.zip --symbol SOLUSDT

Ce script :
  1. Charge le modèle PPO entraîné
  2. Se connecte à Binance (API publique, sans clé)
  3. Attend la fermeture de chaque bougie HTF (15m)
  4. Demande une décision à l'agent
  5. Exécute les trades en papier (simulation uniquement, 0 fonds réels)
  6. Affiche un dashboard terminal en temps réel
  7. Sauvegarde l'historique dans paper_trades.json

Arrêt propre : [q] dans le dashboard, ou Ctrl+C dans le terminal.
"""

import argparse
import curses
import time
import sys
import os
import threading
from datetime import datetime

# Les imports du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_fetcher import LiveDataFeed
from paper_engine import PaperTradingEngine
# dashboard importé plus tard seulement si le dashboard est activé


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOL  = "BTCUSDT"
DEFAULT_HTF     = "15m"
DEFAULT_LTF     = "1m"
REFRESH_INTERVAL = 5   # secondes entre chaque vérification de données
PRICE_REFRESH    = 10    # secondes entre chaque mise à jour du prix affiché


# ──────────────────────────────────────────────────────────────────────────────
# Thread de collecte de données (tourne en arrière-plan)
# ──────────────────────────────────────────────────────────────────────────────

class DataThread(threading.Thread):
    """Thread qui rafraîchit les données Binance toutes les N secondes."""

    def __init__(self, feed: LiveDataFeed, engine: PaperTradingEngine, interval: int = 15):
        super().__init__(daemon=True)
        self.feed = feed
        self.engine = engine
        self.interval = interval
        self.running = True
        self.last_result = {}
        self.status_msg = "En attente de données..."
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            try:
                result = self.feed.refresh()
                price = result.get("current_price", 0.0)

                if result.get("new_htf_candle") and self.feed.is_ready:
                    # Nouvelle bougie HTF clôturée → l'agent décide
                    engine_result = self.engine.on_new_candle(
                        self.feed.htf_candles,
                        self.feed.ltf_candles,
                        price,
                    )
                    with self.lock:
                        self.last_result = engine_result
                        self.status_msg = engine_result.get("message", "Bougie HTF clôturée")

                elif result.get("new_ltf_candle"):
                    # Nouvelle bougie LTF → mise à jour prix seulement
                    with self.lock:
                        self.status_msg = f"Bougie LTF clôturée — prix : {price:.2f}$"

                # Mise à jour du PnL latent de la position ouverte
                if self.engine.open_trade and price > 0:
                    t = self.engine.open_trade
                    if t.direction == "buy":
                        t.pnl_usdt = (price - t.entry_price) * t.position_size
                    else:
                        t.pnl_usdt = (t.entry_price - price) * t.position_size

            except Exception as e:
                with self.lock:
                    self.status_msg = f"Erreur réseau : {e}"
                    self.feed.errors.append(f"{datetime.now().strftime('%H:%M:%S')} {e}")

            time.sleep(self.interval)

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────────────────────────────────────
# Boucle principale (dashboard curses)
# ──────────────────────────────────────────────────────────────────────────────

def run_dashboard(stdscr, feed: LiveDataFeed, engine: PaperTradingEngine, data_thread: DataThread, symbol: str):
    """Boucle principale du dashboard curses."""
    from dashboard import Dashboard, init_colors
    init_colors()
    stdscr.nodelay(True)   # non-bloquant pour la lecture des touches
    stdscr.timeout(500)    # refresh toutes les 500ms

    curses.curs_set(0)     # cache le curseur
    dash = Dashboard(symbol=symbol, htf=feed.htf_interval, ltf=feed.ltf_interval)

    while True:
        # Lecture des touches
        try:
            key = stdscr.getch()
        except Exception:
            key = -1

        if key in (ord('q'), ord('Q')):
            break

        if key in (ord('c'), ord('C')):
            # Clôture manuelle de la position
            if engine.open_trade and feed.current_price > 0:
                engine.force_close(feed.current_price)
                data_thread.status_msg = "✋ Position clôturée manuellement"

        # Mise à jour du dashboard
        with data_thread.lock:
            last_result = dict(data_thread.last_result)
            status_msg  = data_thread.status_msg

        dash.update(
            current_price      = feed.current_price,
            balance            = engine.balance,
            open_trade         = engine.open_trade,
            trades             = engine.trades[:],
            equity_curve       = engine.equity_curve[:],
            stats              = engine.get_stats(),
            last_action        = engine.last_action,
            last_action_time   = engine.last_action_time,
            last_signal        = engine.last_signal,
            last_result        = last_result,
            errors             = feed.errors[-5:],
            status_msg         = status_msg,
            time_to_htf        = feed.time_to_next_htf(),
            time_to_ltf        = feed.time_to_next_ltf(),
            htf_count          = len(feed.htf_candles) if feed.htf_candles is not None else 0,
            ltf_count          = len(feed.ltf_candles) if feed.ltf_candles is not None else 0,
            step               = engine.current_step,
        )
        dash.render(stdscr)

    # Sauvegarde finale à la fermeture
    engine._save_history()


# ──────────────────────────────────────────────────────────────────────────────
# Mode sans dashboard (fallback si terminal non supporté)
# ──────────────────────────────────────────────────────────────────────────────

def run_console(feed: LiveDataFeed, engine: PaperTradingEngine):
    """Mode console simple si curses n'est pas disponible."""
    print("\n🚀 Paper trading démarré (mode console — Ctrl+C pour arrêter)\n")

    while True:
        try:
            result = feed.refresh()
            price = result.get("current_price", 0.0)

            ts = datetime.now().strftime("%H:%M:%S")

            if result.get("new_htf_candle") and feed.is_ready:
                print(f"\n[{ts}] ── Nouvelle bougie HTF ──────────────────────")
                engine_result = engine.on_new_candle(
                    feed.htf_candles, feed.ltf_candles, price
                )
                action = engine_result.get("action_label", "HOLD")
                msg    = engine_result.get("message", "")
                conf   = "✅ CONFORME" if engine_result.get("conformity") else "—"
                print(f"  Prix : {price:.2f}$ | Décision : {action} | {conf}")
                if msg:
                    print(f"  {msg}")
                stats = engine.get_stats()
                print(f"  Balance : {engine.balance:.2f}$ | Trades : {stats['n_trades']} | WR : {stats['win_rate']:.1%}")
            else:
                # Affichage minimal toutes les 30s
                pnl = engine.balance - engine.initial_balance
                print(f"[{ts}] Prix : {price:.2f}$  Balance : {engine.balance:.2f}$  PnL : {pnl:+.2f}$", end="\r")

            time.sleep(REFRESH_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n⏹  Arrêt demandé — sauvegarde...")
            engine._save_history()
            _print_final_report(engine)
            break
        except Exception as e:
            print(f"\n[ERREUR] {e}")
            time.sleep(30)


def _print_final_report(engine: PaperTradingEngine):
    """Rapport final en console."""
    s = engine.get_stats()
    print("\n" + "═" * 55)
    print("  RAPPORT FINAL — PAPER TRADING FVG")
    print("═" * 55)
    print(f"  Balance finale  : {engine.balance:.2f}$")
    print(f"  PnL total       : {s['total_pnl']:>+.2f}$")
    print(f"  Trades          : {s['n_trades']}")
    print(f"  Win rate        : {s['win_rate']:.1%}")
    print(f"  Sharpe ratio    : {s['sharpe']:.3f}")
    print(f"  Max drawdown    : {s['max_drawdown']:.2%}")
    print(f"  Conformité      : {s['conformity']:.1%}")
    print("═" * 55)
    print(f"  Historique sauvegardé : {engine.log_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper trading FVG Base Hits")
    parser.add_argument("--model",     type=str, required=True, help="Chemin vers le modele .zip")
    parser.add_argument("--symbol",    type=str, default=DEFAULT_SYMBOL, help="Symbole Binance (ex: BTCUSDT)")
    parser.add_argument("--htf",       type=str, default=DEFAULT_HTF,    help="Intervalle HTF (defaut: 15m)")
    parser.add_argument("--ltf",       type=str, default=DEFAULT_LTF,    help="Intervalle LTF (defaut: 5m)")
    parser.add_argument("--balance",   type=float, default=10_000.0,     help="Capital paper (defaut: 10000)")
    parser.add_argument("--risk",      type=float, default=0.01,         help="Risque par trade (ex: 0.01 = 1%%)")
    parser.add_argument("--log",       type=str, default=None,           help="Fichier de sauvegarde (defaut: paper_sessions/SYMBOL/SYMBOL.json)")
    parser.add_argument("--leverage",  type=int, default=1,              help="Levier (defaut: 1 = pas de levier, ex: 10 = x10)")
    parser.add_argument("--no_dashboard", action="store_true",           help="Mode console (sans curses)")
    args = parser.parse_args()

    # Isolation par symbole : chaque actif a son propre dossier et fichier JSON.
    # Deux terminaux sur BTCUSDT et SOLUSDT n'interferent jamais.
    symbol_clean = args.symbol.upper().replace("/", "")
    session_dir  = os.path.join("paper_sessions", symbol_clean)
    os.makedirs(session_dir, exist_ok=True)
    log_path = args.log if args.log else os.path.join(session_dir, f"{symbol_clean}.json")

    print("=" * 55)
    print(f"  FVG BASE HITS — PAPER TRADING [{symbol_clean}]")
    print("=" * 55)
    print(f"  Symbole  : {args.symbol}")
    print(f"  HTF : {args.htf}  |  LTF : {args.ltf}")
    print(f"  Capital  : {args.balance:.2f}$  |  Risque : {args.risk*100:.1f}%")
    lev_label = f"x{args.leverage}" if args.leverage > 1 else "Aucun (x1)"
    print(f"  Levier   : {lev_label}")
    print(f"  Modele   : {args.model}")
    print(f"  Session  : {session_dir}/")
    print(f"  Log      : {log_path}")
    print()

    # 1. Chargement du moteur (modele + historique propre a ce symbole)
    print("Chargement du modele PPO...")
    engine = PaperTradingEngine(
        model_path=args.model,
        initial_balance=args.balance,
        risk_pct=args.risk,
        log_path=log_path,
        leverage=args.leverage,
    )
    if not engine.model_loaded:
        print(f"❌ Impossible de charger le modèle : {engine.model_error}")
        print("   Vérifie le chemin et que stable-baselines3 est installé.")
        sys.exit(1)
    print("✅ Modèle chargé.")

    # 2. Connexion Binance et chargement initial des données
    print(f"⏳ Connexion à Binance ({args.symbol})...")
    feed = LiveDataFeed(
        symbol=args.symbol,
        htf=args.htf,
        ltf=args.ltf,
        htf_lookback=150,
        ltf_lookback=450,
    )
    if not feed.initialize():
        print(f"❌ Impossible de récupérer les données Binance.")
        for err in feed.errors:
            print(f"   {err}")
        print("\n   Vérifie ta connexion internet.")
        sys.exit(1)
    print(f"✅ Données chargées — {len(feed.htf_candles)} bougies HTF / {len(feed.ltf_candles)} bougies LTF")

    # Reprise si historique existant
    if engine.trades:
        print(f"📂 Reprise : {len(engine.trades)} trades existants chargés (balance : {engine.balance:.2f}$)")

    print("\n🚀 Lancement...\n")
    time.sleep(1)

    # 3. Thread de données en arrière-plan
    data_thread = DataThread(feed, engine, interval=REFRESH_INTERVAL)
    data_thread.start()

    # 4. Dashboard ou mode console
    if args.no_dashboard:
        run_console(feed, engine)
    else:
        try:
            curses.wrapper(run_dashboard, feed, engine, data_thread, args.symbol)
        except Exception as e:
            print(f"\n⚠️  Dashboard curses non disponible ({e})")
            print("   Basculement en mode console...\n")
            run_console(feed, engine)
        finally:
            data_thread.stop()
            engine._save_history()
            _print_final_report(engine)


if __name__ == "__main__":
    main()