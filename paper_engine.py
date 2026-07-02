"""
paper_engine.py — Moteur de paper trading pour l'agent FVG Base Hits

Gère :
  - L'exécution simulée des ordres (BUY / SELL)
  - Le suivi des positions ouvertes (SL/TP automatique)
  - L'historique des trades et les métriques de performance
  - La persistance des trades dans un fichier JSON
"""

import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from pathlib import Path

from strategy import (
    compute_state_features,
    scan_active_fvgs,
    detect_entry_model_ltf,
)


# ──────────────────────────────────────────────────────────────────────────────
# Structures de données
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveTrade:
    id: int
    direction: str          # 'buy' ou 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float    # en unités de l'actif (ex : 0.001 BTC)
    risk_amount: float      # montant risqué en USDT
    entry_time: str
    had_fvg: bool
    had_ltf_model: bool
    # Rempli à la clôture
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"    # 'open', 'tp_hit', 'sl_hit', 'manual'
    close_reason: str = ""
    leverage: int = 1            # levier appliqué à ce trade
    notional_value: float = 0.0  # valeur notionnelle réelle (position_size × entry_price × leverage)
    margin_used: float = 0.0     # capital immobilisé = notional / leverage


@dataclass
class EngineState:
    balance: float
    initial_balance: float
    open_trade: Optional[LiveTrade]
    trades: list
    equity_curve: list
    last_action: str
    last_action_time: str
    last_signal: str
    conformity_ok: bool


# ──────────────────────────────────────────────────────────────────────────────
# Moteur principal
# ──────────────────────────────────────────────────────────────────────────────

class PaperTradingEngine:
    """
    Moteur de paper trading qui exécute l'agent FVG en temps réel.

    À chaque nouvelle bougie HTF clôturée :
      1. Calcule l'observation (vecteur de 21 features)
      2. Demande une décision à l'agent PPO
      3. Vérifie la conformité stratégie (FVG + LTF)
      4. Exécute ou refuse le trade en papier
      5. Vérifie le SL/TP de la position ouverte
    """

    def __init__(
        self,
        model_path: str,
        initial_balance: float = 10_000.0,
        risk_pct: float = 0.01,
        sl_atr_mult: float = 1.0,
        log_path: str = "paper_trades.json",
        min_fvg_conformity: bool = True,   # bloquer si pas de FVG (comme l'env)
        leverage: int = 1,                 # levier (1 = pas de levier, 10 = x10)
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.sl_atr_mult = sl_atr_mult
        self.leverage = max(1, int(leverage))  # minimum 1x
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_fvg_conformity = min_fvg_conformity

        self.open_trade: Optional[LiveTrade] = None
        self.trades: list[LiveTrade] = []
        self.equity_curve: list[float] = [initial_balance]
        self.trade_counter = 0

        # Cooldown : pas de re-trade dans les 3 bougies suivant une clôture
        self.last_close_step = -10
        self.current_step = 0

        # Historique des décisions pour affichage
        self.last_action = "—"
        self.last_action_time = "—"
        self.last_signal = "—"
        self.last_obs: Optional[np.ndarray] = None
        self.conformity_ok = False

        # Chargement du modèle PPO
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            self.model_loaded = True
        except Exception as e:
            self.model = None
            self.model_loaded = False
            self.model_error = str(e)

        # Chargement de l'historique si le fichier existe
        self._load_history()

    # ------------------------------------------------------------------
    # Méthode principale appelée à chaque nouvelle bougie HTF
    # ------------------------------------------------------------------

    def on_new_candle(
        self,
        htf_candles: pd.DataFrame,
        ltf_candles: pd.DataFrame,
        current_price: float,
    ) -> dict:
        """
        Point d'entrée principal.
        Appelé quand une nouvelle bougie HTF se ferme.
        Retourne un dict décrivant ce qui s'est passé.
        """
        self.current_step += 1
        result = {
            "step": self.current_step,
            "action": 0,
            "action_label": "HOLD",
            "trade_opened": None,
            "trade_closed": None,
            "conformity": False,
            "fvg_active": False,
            "ltf_signal": False,
            "balance": self.balance,
            "message": "",
        }

        if not self.model_loaded:
            result["message"] = f"Modèle non chargé : {getattr(self, 'model_error', '?')}"
            return result

        htf_reset = htf_candles.copy().reset_index(drop=False)
        ltf_reset = ltf_candles.copy().reset_index(drop=False)

        htf_idx = len(htf_reset) - 1
        ltf_idx = len(ltf_reset) - 1

        # 1. Vérifier SL/TP sur la position ouverte
        if self.open_trade:
            closed = self._check_sl_tp(htf_candles, current_price)
            if closed:
                result["trade_closed"] = self.open_trade
                result["message"] = (
                    f"{'✅ TP' if self.open_trade.status == 'tp_hit' else '❌ SL'} "
                    f"atteint — PnL : {self.open_trade.pnl_usdt:+.2f}$"
                )
                self.last_close_step = self.current_step
                self.open_trade = None
                self._save_history()

        # 2. Construire l'observation pour l'agent
        if htf_idx < 5 or ltf_idx < 5:
            result["message"] = "Pas assez de données"
            return result

        # Portfolio features (miroir de env.py)
        strategy_feats = compute_state_features(htf_reset, ltf_reset, htf_idx, ltf_idx)

        if self.open_trade:
            if self.open_trade.direction == "buy":
                latent = (current_price - self.open_trade.entry_price) / self.open_trade.entry_price
            else:
                latent = (self.open_trade.entry_price - current_price) / self.open_trade.entry_price
            portfolio = np.array([
                1.0,
                1.0 if self.open_trade.direction == "buy" else -1.0,
                np.clip(latent * 10, -1, 1),
                self.balance / self.initial_balance - 1.0,
            ], dtype=np.float32)
        else:
            portfolio = np.array([
                0.0, 0.0, 0.0,
                self.balance / self.initial_balance - 1.0,
            ], dtype=np.float32)

        obs = np.concatenate([strategy_feats, portfolio]).astype(np.float32)
        self.last_obs = obs

        # 3. Décision de l'agent
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        result["action"] = action
        result["action_label"] = ["HOLD", "BUY", "SELL"][action]

        # 4. Analyse de conformité stratégie
        active_fvgs = scan_active_fvgs(htf_reset, htf_idx, lookback=10)
        direction = "buy" if action == 1 else "sell"

        relevant_fvgs = []
        if action != 0:
            relevant_fvgs = [
                f for f in active_fvgs
                if (direction == "buy"  and f.direction == "bearish" and current_price <= f.top)
                or (direction == "sell" and f.direction == "bullish" and current_price >= f.bottom)
            ]

        has_fvg = len(relevant_fvgs) > 0
        ltf_score = detect_entry_model_ltf(ltf_reset, ltf_idx, direction) if action != 0 else 0.0
        has_ltf = ltf_score >= 0.5
        conformity = has_fvg and has_ltf

        result["fvg_active"] = len(active_fvgs) > 0
        result["ltf_signal"] = has_ltf
        result["conformity"] = conformity
        self.conformity_ok = conformity

        # Signal lisible
        fvg_dirs = list(set(f.direction for f in active_fvgs))
        self.last_signal = (
            f"FVG {'+'.join(fvg_dirs)} | LTF {'✓' if has_ltf else '✗'}"
            if active_fvgs else "Pas de FVG actif"
        )

        # 5. Exécution du trade (si action non nulle)
        cooldown_ok = (self.current_step - self.last_close_step) >= 3

        if action != 0 and self.open_trade is None and cooldown_ok:
            if self.min_fvg_conformity and not has_fvg:
                result["message"] = "🚫 Trade bloqué — aucun FVG actif (règle stratégie)"
                self.last_action = f"BLOQUÉ ({['HOLD','BUY','SELL'][action]})"
                self.last_action_time = datetime.now().strftime("%H:%M:%S")
            else:
                trade = self._open_trade(
                    direction, current_price,
                    htf_candles, ltf_candles, ltf_idx,
                    has_fvg, has_ltf,
                )
                if trade:
                    result["trade_opened"] = trade
                    result["message"] = (
                        f"{'📈' if direction == 'buy' else '📉'} "
                        f"{direction.upper()} ouvert @ {current_price:.2f}$ "
                        f"| SL: {trade.stop_loss:.2f} | TP: {trade.take_profit:.2f}"
                        f"{' ✅ CONFORME' if conformity else ' ⚠️ FVG seul'}"
                    )
                    self.last_action = f"{'BUY' if direction == 'buy' else 'SELL'} @ {current_price:.2f}"
                    self.last_action_time = datetime.now().strftime("%H:%M:%S")
        elif action != 0 and not cooldown_ok:
            result["message"] = f"⏳ Cooldown actif ({3 - (self.current_step - self.last_close_step)} bougies)"
        elif action == 0:
            self.last_action = "HOLD"
            self.last_action_time = datetime.now().strftime("%H:%M:%S")

        self.equity_curve.append(self.balance)
        result["balance"] = self.balance
        return result

    # ------------------------------------------------------------------
    # Gestion des trades
    # ------------------------------------------------------------------

    def _open_trade(
        self,
        direction: str,
        current_price: float,
        htf_candles: pd.DataFrame,
        ltf_candles: pd.DataFrame,
        ltf_idx: int,
        had_fvg: bool,
        had_ltf: bool,
    ) -> Optional[LiveTrade]:
        """
        Ouvre un trade papier.

        Logique SL/TP :
          1. SL structurel  : low/high de la bougie HTF courante (règle stratégie)
          2. SL ATR HTF     : ATR des 14 dernières bougies HTF x multiplicateur
          3. On prend le MAX des deux distances (SL le plus éloigné = plus prudent)
          4. Plancher absolu : SL minimum = 0.15% du prix (évite les SL ridicules)
          5. Plafond absolu  : SL maximum = 1.5% du prix  (évite les SL trop larges)
          6. TP = distance SL exacte (ratio 1:1, comme dans l'entraînement)
          7. Position size  = risk_amount / distance_SL  (risque fixe en USDT)
        """
        htf_last = htf_candles.iloc[-1]

        # ── 1. SL structurel (mèche de la dernière bougie HTF) ──────────
        if direction == "buy":
            sl_structural = current_price - float(htf_last["low"])
        else:
            sl_structural = float(htf_last["high"]) - current_price
        sl_structural = abs(sl_structural)

        # ── 2. ATR HTF sur 14 bougies ────────────────────────────────────
        htf_window = htf_candles.iloc[-14:] if len(htf_candles) >= 14 else htf_candles
        atr_htf = float((htf_window["high"] - htf_window["low"]).mean())
        sl_atr = atr_htf * self.sl_atr_mult

        # ── 3. On retient la distance la plus significative ──────────────
        sl_dist = max(sl_structural, sl_atr)

        # ── 4 & 5. Plancher et plafond en % du prix ─────────────────────
        sl_min = current_price * 0.0015   # 0.15% minimum
        sl_max = current_price * 0.015    # 1.5%  maximum
        sl_dist = max(sl_dist, sl_min)
        sl_dist = min(sl_dist, sl_max)

        # ── 6. Calcul final SL / TP ──────────────────────────────────────
        if direction == "buy":
            stop_loss   = current_price - sl_dist
            take_profit = current_price + sl_dist   # ratio 1:1
        else:
            stop_loss   = current_price + sl_dist
            take_profit = current_price - sl_dist   # ratio 1:1

        risk = sl_dist  # distance en prix = risque par unité

        # ── 7. Sizing : risque fixe en USDT ─────────────────────────────
        # position_size = combien d'unités pour que (risk * position_size) = risk_amount
        # Ex: risk=100$, risk_amount=50$ → position_size=0.5 BTC
        #     PnL si TP : 0.5 BTC × 100$ = +50$  ✅
        #     PnL si SL : 0.5 BTC × 100$ = -50$  ✅
        # ── 8. Levier ────────────────────────────────────────────────────
        # Sans levier : position_size = risk_amount / risk
        # Avec levier x10 : position_size × 10
        #   → même SL en % MAIS le PnL est multiplié par 10
        #   → si SL = 1%, perte = 10% du capital engagé (pas du capital total)
        #   → le risk_amount reste le même (1% du capital)
        #     MAIS la position notionnelle est 10x plus grande
        #
        # Concrètement : margin_used = risk_amount / risk * entry_price / leverage
        # = le capital réellement immobilisé pour ouvrir la position
        risk_amount    = self.balance * self.risk_pct
        position_size  = (risk_amount / risk) * self.leverage   # taille × levier
        notional_value = position_size * current_price           # exposition réelle
        margin_used    = notional_value / self.leverage          # capital immobilisé

        # Sécurité : vérifier que la marge disponible est suffisante
        if margin_used > self.balance * 0.9:
            # On réduit la taille pour ne pas dépasser 90% du capital en marge
            position_size  = (self.balance * 0.9 * self.leverage) / current_price
            notional_value = position_size * current_price
            margin_used    = notional_value / self.leverage

        self.trade_counter += 1
        trade = LiveTrade(
            id=self.trade_counter,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            had_fvg=had_fvg,
            had_ltf_model=had_ltf,
            leverage=self.leverage,
            notional_value=notional_value,
            margin_used=margin_used,
        )
        # Log lisible dans la console
        sl_pct = abs(current_price - stop_loss) / current_price * 100
        lev_str = f" | Levier: x{self.leverage}" if self.leverage > 1 else ""
        print(
            f"  [TRADE #{self.trade_counter}] {direction.upper()} @ {current_price:.4f}"
            f" | SL: {stop_loss:.4f} ({sl_pct:.2f}%)"
            f" | TP: {take_profit:.4f}"
            f" | Risk: {risk_amount:.2f}$ | Size: {position_size:.6f}"
            f" | Notionnel: {notional_value:.2f}${lev_str}"
        )
        self.open_trade = trade
        return trade

    def _check_sl_tp(self, htf_candles: pd.DataFrame, current_price: float) -> bool:
        """Vérifie si SL ou TP est atteint. Retourne True si trade clôturé."""
        if not self.open_trade:
            return False

        t = self.open_trade
        candle = htf_candles.iloc[-1]

        hit_price = None
        reason = ""

        if t.direction == "buy":
            if float(candle["high"]) >= t.take_profit:
                hit_price, reason = t.take_profit, "tp_hit"
            elif float(candle["low"]) <= t.stop_loss:
                hit_price, reason = t.stop_loss, "sl_hit"
        else:
            if float(candle["low"]) <= t.take_profit:
                hit_price, reason = t.take_profit, "tp_hit"
            elif float(candle["high"]) >= t.stop_loss:
                hit_price, reason = t.stop_loss, "sl_hit"

        if hit_price is None:
            # Mise à jour du PnL latent (non réalisé)
            if t.direction == "buy":
                t.pnl_usdt = (current_price - t.entry_price) * t.position_size
            else:
                t.pnl_usdt = (t.entry_price - current_price) * t.position_size
            return False

        # Clôture
        if t.direction == "buy":
            pnl = (hit_price - t.entry_price) * t.position_size
        else:
            pnl = (t.entry_price - hit_price) * t.position_size

        t.exit_price = hit_price
        t.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t.pnl_usdt = pnl
        t.pnl_pct = pnl / self.balance * 100
        t.status = reason
        t.close_reason = "TP atteint" if reason == "tp_hit" else "SL atteint"

        self.balance += pnl
        self.trades.append(t)
        return True

    def force_close(self, current_price: float):
        """Clôture manuelle de la position ouverte."""
        if not self.open_trade:
            return
        t = self.open_trade
        if t.direction == "buy":
            pnl = (current_price - t.entry_price) * t.position_size
        else:
            pnl = (t.entry_price - current_price) * t.position_size
        t.exit_price = current_price
        t.exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t.pnl_usdt = pnl
        t.pnl_pct = pnl / self.balance * 100
        t.status = "manual"
        t.close_reason = "Clôture manuelle"
        self.balance += pnl
        self.trades.append(t)
        self.open_trade = None
        self._save_history()

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Retourne les métriques de performance actuelles."""
        all_trades = self.trades[:]
        if self.open_trade:
            # PnL latent de la position ouverte inclus dans l'affichage
            pass

        if not all_trades:
            return {
                "n_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "max_drawdown": 0.0, "sharpe": 0.0,
                "conformity": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "balance": self.balance,
            }

        pnls = [t.pnl_usdt for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        eq = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max) / np.maximum(running_max, 1)
        max_dd = float(dd.min()) if len(dd) > 1 else 0.0

        sharpe = 0.0
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)

        conf_trades = [t for t in all_trades if t.had_fvg and t.had_ltf_model]
        conformity = len(conf_trades) / len(all_trades) if all_trades else 0.0

        return {
            "n_trades":    len(all_trades),
            "win_rate":    len(wins) / len(pnls) if pnls else 0.0,
            "total_pnl":   sum(pnls),
            "max_drawdown": max_dd,
            "sharpe":      sharpe,
            "conformity":  conformity,
            "avg_win":     np.mean(wins) if wins else 0.0,
            "avg_loss":    np.mean(losses) if losses else 0.0,
            "balance":     self.balance,
        }

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def _save_history(self):
        """Sauvegarde l'historique des trades en JSON."""
        try:
            data = {
                "initial_balance": self.initial_balance,
                "balance": self.balance,
                "trades": [asdict(t) for t in self.trades],
                "equity_curve": self.equity_curve[-500:],  # limite la taille
                "saved_at": datetime.now().isoformat(),
            }
            self.log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            pass

    def _load_history(self):
        """Recharge l'historique si le fichier existe (reprise après interruption)."""
        if not self.log_path.exists():
            return
        try:
            data = json.loads(self.log_path.read_text())
            self.balance = data.get("balance", self.initial_balance)
            self.equity_curve = data.get("equity_curve", [self.initial_balance])
            raw_trades = data.get("trades", [])
            self.trades = [LiveTrade(**t) for t in raw_trades]
            self.trade_counter = len(self.trades)
        except Exception:
            pass  # Fichier corrompu → repart de zéro