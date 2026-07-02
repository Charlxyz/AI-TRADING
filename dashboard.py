"""
dashboard.py — Dashboard terminal temps réel pour le paper trading FVG

Affiche en temps réel :
  ┌─ En-tête : symbole, prix, balance, PnL global
  ├─ Position ouverte : direction, entry, SL, TP, PnL latent
  ├─ Signal actuel : FVG, LTF, décision agent, conformité
  ├─ Métriques : win rate, Sharpe, drawdown, nb trades
  ├─ Mini courbe d'équité ASCII
  └─ Historique des 10 derniers trades
"""

import curses
import time
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Couleurs (curses)
# ──────────────────────────────────────────────────────────────────────────────

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN,   -1)  # gains / BUY / OK
    curses.init_pair(2, curses.COLOR_RED,     -1)  # pertes / SELL / KO
    curses.init_pair(3, curses.COLOR_CYAN,    -1)  # titres / info
    curses.init_pair(4, curses.COLOR_YELLOW,  -1)  # alertes / latent
    curses.init_pair(5, curses.COLOR_WHITE,   -1)  # texte normal
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # conformité / signal
    curses.init_pair(7, curses.COLOR_BLACK,   curses.COLOR_WHITE)  # header inversé

GREEN   = curses.color_pair(1)
RED     = curses.color_pair(2)
CYAN    = curses.color_pair(3)
YELLOW  = curses.color_pair(4)
NORMAL  = curses.color_pair(5)
MAGENTA = curses.color_pair(6)
HEADER  = curses.color_pair(7)
BOLD    = curses.A_BOLD


# ──────────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ──────────────────────────────────────────────────────────────────────────────

def safe_addstr(win, y, x, text, attr=0):
    """addstr sans crasher si hors limites."""
    try:
        h, w = win.getmaxyx()
        if y < h and x < w:
            win.addstr(y, x, text[:w - x - 1], attr)
    except curses.error:
        pass


def draw_hline(win, y, w, char="─"):
    safe_addstr(win, y, 0, char * (w - 1))


def draw_box_title(win, y, title, w):
    """Ligne de titre de section."""
    line = f"┌── {title} " + "─" * max(0, w - len(title) - 6) + "┐"
    safe_addstr(win, y, 0, line, CYAN | BOLD)


def mini_equity_chart(equity: list, width: int = 50, height: int = 5) -> list[str]:
    """
    Génère une mini courbe d'équité en ASCII sur `height` lignes.
    Retourne une liste de strings.
    """
    if len(equity) < 2:
        return [" " * width] * height

    vals = equity[-width:]  # dernières N valeurs
    mn, mx = min(vals), max(vals)
    spread = mx - mn if mx != mn else 1.0

    # Normalise entre 0 et height-1
    normalized = [int((v - mn) / spread * (height - 1)) for v in vals]

    lines = []
    for row in range(height - 1, -1, -1):
        line = ""
        for i, nv in enumerate(normalized):
            if nv == row:
                line += "●"
            elif i > 0 and (
                (normalized[i-1] < row <= nv) or (normalized[i-1] > row >= nv)
            ):
                line += "│"
            else:
                line += " "
        # Padding à droite
        line = line.ljust(width)
        lines.append(line)

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Classe Dashboard
# ──────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """Dashboard terminal pour le paper trading."""

    def __init__(self, symbol: str = "BTCUSDT", htf: str = "15m", ltf: str = "5m"):
        self.symbol = symbol
        self.htf = htf
        self.ltf = ltf
        self.stdscr = None

        # Données live (mises à jour depuis le moteur principal)
        self.current_price: float = 0.0
        self.balance: float = 10_000.0
        self.initial_balance: float = 10_000.0
        self.open_trade = None
        self.trades: list = []
        self.equity_curve: list = [10_000.0]
        self.stats: dict = {}
        self.last_action: str = "—"
        self.last_action_time: str = "—"
        self.last_signal: str = "—"
        self.last_result: dict = {}
        self.errors: list = []
        self.status_msg: str = "Initialisation..."
        self.time_to_htf: float = 0.0
        self.time_to_ltf: float = 0.0
        self.htf_candles_count: int = 0
        self.ltf_candles_count: int = 0
        self.step: int = 0

    def update(
        self,
        current_price: float,
        balance: float,
        open_trade,
        trades: list,
        equity_curve: list,
        stats: dict,
        last_action: str,
        last_action_time: str,
        last_signal: str,
        last_result: dict,
        errors: list,
        status_msg: str,
        time_to_htf: float,
        time_to_ltf: float,
        htf_count: int,
        ltf_count: int,
        step: int,
    ):
        """Met à jour toutes les données affichées."""
        self.current_price = current_price
        self.balance = balance
        self.open_trade = open_trade
        self.trades = trades
        self.equity_curve = equity_curve
        self.stats = stats
        self.last_action = last_action
        self.last_action_time = last_action_time
        self.last_signal = last_signal
        self.last_result = last_result
        self.errors = errors
        self.status_msg = status_msg
        self.time_to_htf = time_to_htf
        self.time_to_ltf = time_to_ltf
        self.htf_candles_count = htf_count
        self.ltf_candles_count = ltf_count
        self.step = step

    def render(self, stdscr):
        """Rendu complet du dashboard."""
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        row = 0

        # ── HEADER ──────────────────────────────────────────────────────
        pnl_total = self.balance - self.initial_balance
        pnl_color = GREEN if pnl_total >= 0 else RED
        header = (
            f" FVG BASE HITS — PAPER TRADING │ {self.symbol} │ "
            f"{datetime.now().strftime('%H:%M:%S')} │ Step #{self.step}"
        )
        safe_addstr(stdscr, row, 0, header.ljust(w - 1), HEADER | BOLD)
        row += 1

        # ── PRIX & BALANCE ───────────────────────────────────────────────
        price_str  = f" Prix : {self.current_price:>12.2f} USDT"
        bal_str    = f"Balance : {self.balance:>10.2f}$"
        pnl_str    = f"PnL : {pnl_total:>+9.2f}$ ({pnl_total/self.initial_balance*100:>+.2f}%)"
        next_htf   = f"Prochaine HTF : {int(self.time_to_htf//60):02d}:{int(self.time_to_htf%60):02d}"

        safe_addstr(stdscr, row, 0,        price_str, BOLD)
        safe_addstr(stdscr, row, 28,       bal_str, BOLD)
        safe_addstr(stdscr, row, 52,       pnl_str, pnl_color | BOLD)
        safe_addstr(stdscr, row, 80,       next_htf, YELLOW)
        row += 1
        draw_hline(stdscr, row, w)
        row += 1

        # ── POSITION OUVERTE ────────────────────────────────────────────
        draw_box_title(stdscr, row, "POSITION OUVERTE", w)
        row += 1

        if self.open_trade:
            t = self.open_trade
            dir_color = GREEN if t.direction == "buy" else RED
            dir_label = "▲ BUY " if t.direction == "buy" else "▼ SELL"

            latent = t.pnl_usdt
            lat_color = GREEN if latent >= 0 else RED

            conf_label = "✅ FVG+LTF" if (t.had_fvg and t.had_ltf_model) else ("⚠️ FVG seul" if t.had_fvg else "❌ hors stratégie")

            safe_addstr(stdscr, row, 2, f"{dir_label}", dir_color | BOLD)
            safe_addstr(stdscr, row, 10, f"Entrée : {t.entry_price:.2f}$", NORMAL)
            safe_addstr(stdscr, row, 32, f"SL : {t.stop_loss:.2f}$", RED)
            safe_addstr(stdscr, row, 50, f"TP : {t.take_profit:.2f}$", GREEN)
            safe_addstr(stdscr, row, 68, f"PnL latent : {latent:>+.2f}$", lat_color | BOLD)
            safe_addstr(stdscr, row, 90, conf_label, MAGENTA)
        else:
            safe_addstr(stdscr, row, 2, "Aucune position ouverte", YELLOW)

        row += 1
        draw_hline(stdscr, row, w)
        row += 1

        # ── SIGNAL & DÉCISION ───────────────────────────────────────────
        draw_box_title(stdscr, row, "SIGNAL AGENT", w)
        row += 1

        action_label = self.last_result.get("action_label", self.last_action)
        action_color = GREEN if "BUY" in action_label else (RED if "SELL" in action_label else YELLOW)
        fvg_ok   = self.last_result.get("fvg_active", False)
        ltf_ok   = self.last_result.get("ltf_signal", False)
        conf_ok  = self.last_result.get("conformity", False)

        safe_addstr(stdscr, row, 2,  f"Décision : {action_label:<6}", action_color | BOLD)
        safe_addstr(stdscr, row, 22, f"@ {self.last_action_time}", NORMAL)
        safe_addstr(stdscr, row, 36, f"FVG : {'✓' if fvg_ok else '✗'}", GREEN if fvg_ok else RED)
        safe_addstr(stdscr, row, 46, f"LTF : {'✓' if ltf_ok else '✗'}", GREEN if ltf_ok else RED)
        safe_addstr(stdscr, row, 56, f"Conformité : {'✅ OUI' if conf_ok else '❌ NON'}", GREEN if conf_ok else RED)
        row += 1
        safe_addstr(stdscr, row, 2,  f"Signal : {self.last_signal}", MAGENTA)
        safe_addstr(stdscr, row, 50, f"HTF : {self.htf_candles_count} bougies | LTF : {self.ltf_candles_count} bougies", NORMAL)
        row += 1

        msg = self.last_result.get("message", self.status_msg)
        if msg:
            safe_addstr(stdscr, row, 2, f"→ {msg}", CYAN)
        row += 1
        draw_hline(stdscr, row, w)
        row += 1

        # ── MÉTRIQUES & COURBE ──────────────────────────────────────────
        # Colonnes : métriques à gauche, courbe à droite
        draw_box_title(stdscr, row, "PERFORMANCES", w)
        row += 1

        s = self.stats
        metrics = [
            ("Trades",       f"{s.get('n_trades', 0)}"),
            ("Win rate",     f"{s.get('win_rate', 0):.1%}"),
            ("PnL total",    f"{s.get('total_pnl', 0):>+.2f}$"),
            ("Max drawdown", f"{s.get('max_drawdown', 0):.2%}"),
            ("Sharpe",       f"{s.get('sharpe', 0):.3f}"),
            ("Conformité",   f"{s.get('conformity', 0):.1%}"),
            ("Moy. gain",    f"{s.get('avg_win', 0):>+.2f}$"),
            ("Moy. perte",   f"{s.get('avg_loss', 0):>+.2f}$"),
        ]

        metric_col_w = 28
        chart_start  = metric_col_w + 4
        chart_w      = min(w - chart_start - 2, 60)
        chart_h      = min(len(metrics), 6)

        for i, (label, val) in enumerate(metrics):
            color = NORMAL
            if "PnL" in label or "gain" in label:
                color = GREEN if "+" in val else RED
            elif "perte" in label:
                color = RED
            elif "drawdown" in label:
                color = RED if float(s.get('max_drawdown', 0)) < -0.05 else YELLOW
            elif "Win" in label:
                wr = s.get('win_rate', 0)
                color = GREEN if wr >= 0.5 else RED
            safe_addstr(stdscr, row + i, 2, f"{label:<14} {val:>10}", color)

        # Mini courbe
        if chart_w > 10:
            chart_lines = mini_equity_chart(self.equity_curve, width=chart_w, height=chart_h)
            mn = min(self.equity_curve[-chart_w:] if len(self.equity_curve) > chart_w else self.equity_curve)
            mx = max(self.equity_curve[-chart_w:] if len(self.equity_curve) > chart_w else self.equity_curve)
            safe_addstr(stdscr, row, chart_start, f"Équité  [{mn:.0f}$ — {mx:.0f}$]", CYAN)
            for i, line in enumerate(chart_lines):
                color = GREEN if self.equity_curve[-1] >= self.initial_balance else RED
                safe_addstr(stdscr, row + 1 + i, chart_start, line, color)

        row += max(len(metrics), chart_h) + 2
        if row >= h - 6:
            # Terminal trop petit → skip la suite
            safe_addstr(stdscr, h-1, 0, " [q] Quitter | [c] Clôturer position | Terminal trop petit pour l'historique ", HEADER)
            stdscr.refresh()
            return

        draw_hline(stdscr, row, w)
        row += 1

        # ── HISTORIQUE DES TRADES ────────────────────────────────────────
        draw_box_title(stdscr, row, "HISTORIQUE (10 derniers trades)", w)
        row += 1

        if not self.trades:
            safe_addstr(stdscr, row, 2, "Aucun trade clôturé pour l'instant.", YELLOW)
            row += 1
        else:
            header_t = f"  {'#':>3}  {'Dir':<5}  {'Entrée':>10}  {'Sortie':>10}  {'PnL':>9}  {'%':>6}  {'Statut':<12}  {'Conform.'}"
            safe_addstr(stdscr, row, 0, header_t, CYAN | BOLD)
            row += 1

            for t in reversed(self.trades[-10:]):
                if row >= h - 2:
                    break
                color = GREEN if t.pnl_usdt > 0 else RED
                dir_sym = "▲" if t.direction == "buy" else "▼"
                status  = {"tp_hit": "✅ TP", "sl_hit": "❌ SL", "manual": "✋ Manuel"}.get(t.status, t.status)
                conf    = "✅" if (t.had_fvg and t.had_ltf_model) else ("⚠️" if t.had_fvg else "❌")
                line = (
                    f"  {t.id:>3}  {dir_sym} {t.direction:<4}  "
                    f"{t.entry_price:>10.2f}  {t.exit_price:>10.2f}  "
                    f"{t.pnl_usdt:>+9.2f}$  {t.pnl_pct:>+5.2f}%  "
                    f"{status:<12}  {conf}"
                )
                safe_addstr(stdscr, row, 0, line, color)
                row += 1

        # ── ERREURS ─────────────────────────────────────────────────────
        if self.errors and row < h - 3:
            draw_hline(stdscr, row, w)
            row += 1
            safe_addstr(stdscr, row, 0, f" ⚠ Dernière erreur : {self.errors[-1]}", RED)
            row += 1

        # ── BARRE DE COMMANDES ──────────────────────────────────────────
        cmd_bar = " [q] Quitter  [c] Clôturer position  [r] Rafraîchir  │  Paper Trading — Aucun fonds réel "
        safe_addstr(stdscr, h - 1, 0, cmd_bar.ljust(w - 1), HEADER)

        stdscr.refresh()