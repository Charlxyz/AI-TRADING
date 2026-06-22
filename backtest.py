"""
backtest.py — Compare l'agent RL vs la stratégie pure rule-based

Usage :
    python backtest.py --htf_data data/EURUSD_15m.csv --model models/best_model.zip

Ce script :
  1. Exécute la stratégie pure (règles seules, sans IA) → baseline
  2. Exécute l'agent RL entraîné
  3. Compare les métriques et trace les courbes d'équité
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from env import FVGTradingEnv, Trade
from strategy import scan_active_fvgs, detect_entry_model_ltf
from train import load_candles, generate_ltf_from_htf


# ──────────────────────────────────────────────────────────────────────────────
# Stratégie pure (rule-based) — baseline de comparaison
# ──────────────────────────────────────────────────────────────────────────────

def run_pure_strategy(
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    initial_balance: float = 10_000.0,
    risk_pct: float = 0.01,
) -> dict:
    """
    Exécute la stratégie Base Hits en mode purement algorithmique :
    entre uniquement quand toutes les conditions sont réunies (FVG + LTF model).
    Pas d'apprentissage — c'est le baseline à battre.
    """
    env = FVGTradingEnv(htf, ltf, initial_balance=initial_balance, risk_pct=risk_pct)
    obs, _ = env.reset()
    done = False

    while not done:
        htf_idx = env.htf_step
        ltf_idx = env.ltf_step
        current_price = float(env.htf.iloc[htf_idx]["close"])

        # Règle pure : cherche un FVG + modèle d'entrée → entre automatiquement
        action = 0  # hold par défaut
        if env.open_trade is None:
            active_fvgs = scan_active_fvgs(env.htf, htf_idx, lookback=10)

            for fvg in active_fvgs:
                if fvg.direction == "bearish" and current_price <= fvg.top:
                    # Signal BUY : FVG bearish au-dessus du prix, pas encore comblé
                    score = detect_entry_model_ltf(env.ltf, ltf_idx, "buy")
                    if score >= 0.5:
                        action = 1
                        break
                elif fvg.direction == "bullish" and current_price >= fvg.bottom:
                    # Signal SELL : FVG bullish en dessous du prix, pas encore comblé
                    score = detect_entry_model_ltf(env.ltf, ltf_idx, "sell")
                    if score >= 0.5:
                        action = 2
                        break

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    stats = env.get_performance_stats()
    stats["equity_curve"] = env.equity_curve
    stats["trades"] = env.trades
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Agent RL
# ──────────────────────────────────────────────────────────────────────────────

def run_rl_agent(
    model_path: str,
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    initial_balance: float = 10_000.0,
    risk_pct: float = 0.01,
) -> dict:
    """Charge et exécute l'agent RL en mode déterministe."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("❌ stable-baselines3 requis. pip install stable-baselines3")
        return {}

    model = PPO.load(model_path)
    env = FVGTradingEnv(htf, ltf, initial_balance=initial_balance, risk_pct=risk_pct)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    stats = env.get_performance_stats()
    stats["equity_curve"] = env.equity_curve
    stats["trades"] = env.trades
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation de la comparaison
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(stats_rb: dict, stats_rl: dict, output_path: str = "results/comparison.png"):
    """Trace la comparaison côte à côte."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    colors = {"rule": "#5B8CDB", "rl": "#E87A5D"}

    # 1. Courbes d'équité
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(stats_rb.get("equity_curve", []), label="Stratégie pure (rule-based)", color=colors["rule"], linewidth=1.5, alpha=0.9)
    ax1.plot(stats_rl.get("equity_curve", []), label="Agent RL (PPO)", color=colors["rl"], linewidth=1.5, alpha=0.9)
    ax1.set_title("Courbes d'équité — Stratégie Base Hits FVG", fontsize=12)
    ax1.set_xlabel("Bougies HTF")
    ax1.set_ylabel("Balance ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    # 2. Win rate et conformité
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ["win_rate", "strategy_conformity", "full_strategy_conformity"]
    labels  = ["Win rate", "Conformité\n(FVG)", "Conformité\n(FVG + LTF)"]
    x = np.arange(len(metrics))
    width = 0.35
    rb_vals = [stats_rb.get(m, 0) for m in metrics]
    rl_vals = [stats_rl.get(m, 0) for m in metrics]
    ax2.bar(x - width/2, rb_vals, width, label="Rule-based", color=colors["rule"], alpha=0.85)
    ax2.bar(x + width/2, rl_vals, width, label="RL Agent",   color=colors["rl"],   alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Win rate & conformité stratégie")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25, axis="y")

    # 3. Métriques financières
    ax3 = fig.add_subplot(gs[1, 1])
    fin_metrics = ["total_pnl", "max_drawdown", "sharpe_ratio"]
    fin_labels  = ["PnL total ($)", "Max drawdown", "Sharpe ratio"]
    rb_fin = [stats_rb.get(m, 0) for m in fin_metrics]
    rl_fin = [stats_rl.get(m, 0) for m in fin_metrics]

    table_data = [
        ["", "Rule-based", "RL Agent"],
        [fin_labels[0], f"{rb_fin[0]:+.2f}", f"{rl_fin[0]:+.2f}"],
        [fin_labels[1], f"{rb_fin[1]:.2%}",  f"{rl_fin[1]:.2%}"],
        [fin_labels[2], f"{rb_fin[2]:.3f}",  f"{rl_fin[2]:.3f}"],
        ["Nb trades",   str(stats_rb.get("n_trades", 0)), str(stats_rl.get("n_trades", 0))],
    ]
    ax3.axis("off")
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax3.set_title("Métriques financières")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📊 Comparaison sauvegardée : {output_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest FVG Base Hits strategy")
    parser.add_argument("--htf_data",  type=str, required=True)
    parser.add_argument("--ltf_data",  type=str, default=None)
    parser.add_argument("--model",     type=str, default=None, help="Chemin vers le .zip du modèle RL")
    parser.add_argument("--balance",   type=float, default=10_000.0)
    parser.add_argument("--risk",      type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Part des données pour le test")
    args = parser.parse_args()

    print("📂 Chargement des données...")
    htf = load_candles(args.htf_data)
    ltf = load_candles(args.ltf_data) if args.ltf_data else generate_ltf_from_htf(htf, ratio=3)

    # Utilise les derniers val_ratio% des données (out-of-sample)
    n = len(htf)
    split = int(n * (1 - args.val_ratio))
    htf_test = htf.iloc[split:].copy()
    ratio = max(1, len(ltf) // len(htf))
    ltf_test = ltf.iloc[split * ratio:].copy()

    print(f"📊 Test set : {len(htf_test)} bougies HTF / {len(ltf_test)} bougies LTF")

    # Run stratégie pure
    print("\n▶  Exécution stratégie rule-based...")
    stats_rb = run_pure_strategy(htf_test, ltf_test, args.balance, args.risk)
    print(f"  Trades: {stats_rb.get('n_trades')} | WR: {stats_rb.get('win_rate', 0):.1%} | PnL: {stats_rb.get('total_pnl', 0):+.2f}")

    # Run agent RL (si disponible)
    stats_rl = {}
    if args.model and os.path.exists(args.model):
        print("\n▶  Exécution agent RL...")
        stats_rl = run_rl_agent(args.model, htf_test, ltf_test, args.balance, args.risk)
        print(f"  Trades: {stats_rl.get('n_trades')} | WR: {stats_rl.get('win_rate', 0):.1%} | PnL: {stats_rl.get('total_pnl', 0):+.2f}")
    else:
        print("\n⚠️  Pas de modèle RL fourni — affichage stratégie pure seulement.")
        stats_rl = stats_rb  # fallback pour l'affichage

    os.makedirs("results", exist_ok=True)
    plot_comparison(stats_rb, stats_rl)