"""
env.py — Environnement Gym pour la stratégie Base Hits (FVG)
Compatible stable-baselines3 / gymnasium.

Actions :
  0 = Ne rien faire (hold)
  1 = Ouvrir un BUY
  2 = Ouvrir un SELL

L'env gère automatiquement :
  - Le SL/TP selon les règles de la stratégie (ratio 1:1)
  - La clôture automatique des positions ouvertes
  - Le reward shaping qui pénalise les trades hors-stratégie
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional
from strategy import compute_state_features, scan_active_fvgs, detect_entry_model_ltf


@dataclass
class Trade:
    direction: str      # 'buy' ou 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_step: int
    exit_price: float = 0.0
    exit_step: int = 0
    pnl: float = 0.0
    closed: bool = False
    # Méta-données pour le reward shaping
    had_fvg_signal: bool = False
    had_entry_model: bool = False


class FVGTradingEnv(gym.Env):
    """
    Environnement de trading pour la stratégie Base Hits.

    Paramètres
    ----------
    htf_candles : DataFrame avec colonnes [open, high, low, close, volume]
                  Index datetime. Représente le 15m ou 1h.
    ltf_candles : DataFrame identique pour le 1m ou 5m.
    initial_balance : Capital de départ (en unités de compte).
    risk_pct : Pourcentage du capital risqué par trade (ex: 0.01 = 1%).
    sl_atr_mult : Multiplicateur ATR pour le stop loss.
    reward_scale : Multiplicateur global du reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        htf_candles: pd.DataFrame,
        ltf_candles: pd.DataFrame,
        initial_balance: float = 10_000.0,
        risk_pct: float = 0.01,
        sl_atr_mult: float = 1.0,
        reward_scale: float = 1.0,
    ):
        super().__init__()

        self.htf = htf_candles.copy().reset_index(drop=False)
        self.ltf = ltf_candles.copy().reset_index(drop=False)
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.sl_atr_mult = sl_atr_mult
        self.reward_scale = reward_scale

        # Aligne l'index LTF sur le HTF (ratio temporel approximatif)
        self._ltf_ratio = max(1, len(self.ltf) // len(self.htf))

        # Espaces gym
        n_features = 17
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(n_features + 4,), dtype=np.float32
        )
        # +4 : [position_ouverte, direction, pnl_latent, balance_normalisée]

        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        self._reset_state()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        current_price = float(self.htf.iloc[self.htf_step]["close"])

        # 1. Gérer la position ouverte (SL/TP automatique)
        if self.open_trade is not None:
            reward += self._check_sl_tp(current_price)
            if self.open_trade and self.open_trade.closed:
                self.open_trade = None

        # 2. Traiter l'action de l'agent
        # Cooldown : interdit de retradre moins de 3 bougies après la clôture du dernier trade
        last_trade_step = self.trades[-1].exit_step if self.trades else 0
        cooldown_ok = (self.htf_step - last_trade_step) >= 3

        if action != 0 and self.open_trade is None and cooldown_ok:
            reward += self._execute_action(action, current_price)
        elif action != 0 and self.open_trade is None and not cooldown_ok:
            reward -= 0.01  # pénalise le sur-trading frénétique
        elif action != 0 and self.open_trade is not None:
            reward -= 0.005 * self.reward_scale
        elif action == 0 and self.open_trade is None:
            # Pénalité légère pour inaction quand un FVG valide est présent
            active_fvgs = scan_active_fvgs(self.htf, self.htf_step, lookback=10)
            if active_fvgs:
                reward -= 0.002

        # 3. Avancer le temps
        self.htf_step += 1
        self.ltf_step = min(self.htf_step * self._ltf_ratio, len(self.ltf) - 1)

        # 4. Conditions de fin d'épisode
        if self.htf_step >= len(self.htf) - 1:
            terminated = True
            # Fermer la position ouverte au prix actuel si épisode terminé
            if self.open_trade and not self.open_trade.closed:
                close_price = float(self.htf.iloc[-1]["close"])
                pnl = self._close_trade(self.open_trade, close_price, self.htf_step)
                reward += pnl * 0.5  # reward réduit pour clôture forcée
                self.open_trade = None

        if self.balance <= self.initial_balance * 0.5:
            terminated = True  # Arrêt si drawdown > 50%

        # 5. Statistiques
        self.equity_curve.append(self.balance)
        info = {
            "balance": self.balance,
            "n_trades": len(self.trades),
            "htf_step": self.htf_step,
        }

        return self._get_obs(), reward * self.reward_scale, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"Step {self.htf_step} | Balance: {self.balance:.2f} "
            f"| Trades: {len(self.trades)} "
            f"| Position: {self.open_trade.direction if self.open_trade else 'none'}"
        )

    # ------------------------------------------------------------------
    # Logique interne
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.htf_step = 5       # commence avec assez de contexte
        self.ltf_step = 5 * self._ltf_ratio if hasattr(self, '_ltf_ratio') else 5
        self.balance = self.initial_balance
        self.open_trade: Optional[Trade] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [self.initial_balance]

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation complet."""
        strategy_features = compute_state_features(
            self.htf, self.ltf, self.htf_step, self.ltf_step
        )

        # Features supplémentaires sur l'état du portefeuille
        if self.open_trade:
            current_price = float(self.htf.iloc[self.htf_step]["close"])
            if self.open_trade.direction == "buy":
                latent_pnl = (current_price - self.open_trade.entry_price) / self.open_trade.entry_price
            else:
                latent_pnl = (self.open_trade.entry_price - current_price) / self.open_trade.entry_price
            portfolio_feats = np.array([
                1.0,
                1.0 if self.open_trade.direction == "buy" else -1.0,
                np.clip(latent_pnl * 10, -1, 1),
                self.balance / self.initial_balance - 1.0,
            ], dtype=np.float32)
        else:
            portfolio_feats = np.array([
                0.0, 0.0, 0.0,
                self.balance / self.initial_balance - 1.0,
            ], dtype=np.float32)

        return np.concatenate([strategy_features, portfolio_feats])

    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Tente d'exécuter un trade. Applique le reward shaping de la stratégie.
        Retourne un reward immédiat (positif si conforme à la stratégie, négatif sinon).
        """
        direction = "buy" if action == 1 else "sell"
        reward = 0.0

        # ── Vérifier la conformité avec la stratégie ──────────────────
        active_fvgs = scan_active_fvgs(self.htf, self.htf_step, lookback=10)
        
        # Pour un BUY : un FVG bearish doit être actif (= au-dessus du prix,
        #   pas encore comblé) ET le prix doit être en train de revenir
        #   dans la zone du gap (pas encore ressorti par le haut).
        # Pour un SELL : symétrique avec un FVG bullish.
        relevant_fvgs = [
            f for f in active_fvgs
            if (direction == "buy" and f.direction == "bearish" and current_price <= f.top)
            or (direction == "sell" and f.direction == "bullish" and current_price >= f.bottom)
        ]
        has_fvg_signal = len(relevant_fvgs) > 0

        # Modèle d'entrée LTF
        entry_score = detect_entry_model_ltf(self.ltf, self.ltf_step, direction)
        has_entry_model = entry_score >= 0.5

        # ── Reward shaping immédiat ────────────────────────────────────
        # RÈGLE DURE : sans FVG, le trade est physiquement bloqué.
        # L'agent reçoit une pénalité et ne peut pas contourner la règle.
        if not has_fvg_signal:
            return -0.03  # pénalité + trade refusé = impossible de tricher

        # FVG présent mais pas de confirmation LTF → on laisse trader,
        # mais avec une pénalité légère (et non plus un bonus) : prendre un
        # trade sans confirmation LTF est un choix sous-optimal, l'agent
        # doit le ressentir dès l'ouverture, pas seulement à la clôture.
        if has_fvg_signal and not has_entry_model:
            reward -= 0.015

        if has_fvg_signal and has_entry_model:
            # Setup complet FVG + Breaker Block : récompense forte immédiate
            reward += 0.08

        # ── Calcul du SL/TP selon les règles de la stratégie ──────────
        atr = self._compute_atr(self.ltf, self.ltf_step, period=5)
        sl_distance = atr * self.sl_atr_mult

        # SL serré sur la mèche de la 3e bougie (règle de la stratégie)
        htf_last = self.htf.iloc[self.htf_step]
        if direction == "buy":
            stop_loss  = htf_last["low"] - sl_distance
            risk       = current_price - stop_loss
            take_profit = current_price + risk  # ratio 1:1
        else:
            stop_loss  = htf_last["high"] + sl_distance
            risk       = stop_loss - current_price
            take_profit = current_price - risk  # ratio 1:1

        if risk <= 0:
            return reward - 0.01

        # ── Sizing de position (risk % du capital) ────────────────────
        risk_amount = self.balance * self.risk_pct
        position_size = risk_amount / risk  # unités

        trade = Trade(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_step=self.htf_step,
            had_fvg_signal=has_fvg_signal,
            had_entry_model=has_entry_model,
        )
        self.open_trade = trade
        return reward

    def _check_sl_tp(self, current_price: float) -> float:
        """Vérifie si le SL ou le TP est atteint sur la bougie actuelle."""
        if not self.open_trade or self.open_trade.closed:
            return 0.0

        t = self.open_trade
        htf_candle = self.htf.iloc[self.htf_step]
        reward = 0.0

        # Vérifie TP puis SL (optimiste : on suppose que le TP peut être touché en premier)
        if t.direction == "buy":
            if htf_candle["high"] >= t.take_profit:
                pnl = self._close_trade(t, t.take_profit, self.htf_step)
                reward = self._pnl_reward(pnl, t)
            elif htf_candle["low"] <= t.stop_loss:
                pnl = self._close_trade(t, t.stop_loss, self.htf_step)
                reward = self._pnl_reward(pnl, t)
        else:
            if htf_candle["low"] <= t.take_profit:
                pnl = self._close_trade(t, t.take_profit, self.htf_step)
                reward = self._pnl_reward(pnl, t)
            elif htf_candle["high"] >= t.stop_loss:
                pnl = self._close_trade(t, t.stop_loss, self.htf_step)
                reward = self._pnl_reward(pnl, t)

        return reward

    def _close_trade(self, trade: Trade, exit_price: float, step: int) -> float:
        """Clôture le trade et met à jour le capital. Retourne le PnL brut."""
        if trade.direction == "buy":
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        risk = abs(trade.entry_price - trade.stop_loss)
        risk_amount = self.balance * self.risk_pct
        position_size = risk_amount / risk if risk > 0 else 0
        pnl = pnl_pct * trade.entry_price * position_size

        self.balance += pnl
        trade.exit_price = exit_price
        trade.exit_step = step
        trade.pnl = pnl
        trade.closed = True
        self.trades.append(trade)
        return pnl

    def _pnl_reward(self, pnl: float, trade: Trade) -> float:
        """
        Convertit le PnL en reward normalisé.
        Bonus si le trade respectait les règles de la stratégie.

        Le multiplicateur ne s'applique qu'aux gains (pnl > 0) : on ne
        veut pas que l'agent soit "moins puni" sur une perte simplement
        parce que le trade était hors-stratégie, ça brouillerait le
        signal de gestion du risque. On veut seulement que les gains
        FVG-seul rapportent structurellement moins que les gains conformes,
        pour que l'agent n'ait pas intérêt à délaisser le LTF même si ces
        trades sont rentables en valeur brute.
        """
        normalized = pnl / (self.initial_balance * self.risk_pct)
        if normalized > 0:
            if trade.had_fvg_signal and trade.had_entry_model:
                normalized *= 1.5   # gains conformes : fortement valorisés
            elif trade.had_fvg_signal:
                normalized *= 0.6   # gains FVG-seul : nettement écrêtés
        else:
            # Sur une perte, légère pénalité supplémentaire si hors-stratégie
            # (renforce le risque perçu de trader sans confirmation LTF)
            if trade.had_fvg_signal and not trade.had_entry_model:
                normalized *= 1.15
        return float(np.clip(normalized, -2.0, 2.0))

    @staticmethod
    def _compute_atr(candles: pd.DataFrame, idx: int, period: int = 5) -> float:
        """ATR simplifié (moyenne des ranges sur `period` bougies)."""
        start = max(0, idx - period)
        window = candles.iloc[start: idx + 1]
        if len(window) == 0:
            return 0.001
        return float((window["high"] - window["low"]).mean())

    # ------------------------------------------------------------------
    # Métriques de performance
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> dict:
        """Retourne les métriques clés de la session."""
        if not self.trades:
            return {
                "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "final_balance": self.balance,
                "max_drawdown": 0.0, "sharpe_ratio": 0.0,
                "strategy_conformity": 0.0, "full_strategy_conformity": 0.0,
            }

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        # Sharpe ratio simplifié (sur les PnL des trades)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Taux de conformité à la stratégie
        strategy_trades = [t for t in self.trades if t.had_fvg_signal]
        full_strategy_trades = [t for t in self.trades if t.had_fvg_signal and t.had_entry_model]

        return {
            "n_trades": len(self.trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "total_pnl": sum(pnls),
            "final_balance": self.balance,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "strategy_conformity": len(strategy_trades) / len(self.trades) if self.trades else 0,
            "full_strategy_conformity": len(full_strategy_trades) / len(self.trades) if self.trades else 0,
        }