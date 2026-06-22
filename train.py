"""
train.py — Entraînement de l'agent PPO sur la stratégie Base Hits (FVG)

Usage :
    python train.py --data_path data/EURUSD_15m.csv --timesteps 500000

Le script :
  1. Charge les données HTF et LTF depuis des CSV
  2. Sépare train (80%) / validation (20%) temporellement
  3. Entraîne un agent PPO avec stable-baselines3
  4. Évalue sur le set de validation
  5. Sauvegarde le modèle et les stats
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
    )
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  stable-baselines3 non installé. Lance : pip install stable-baselines3")

from env import FVGTradingEnv


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ──────────────────────────────────────────────────────────────────────────────

def load_candles(path: str) -> pd.DataFrame:
    """
    Charge un CSV OHLCV.
    Format attendu : datetime, open, high, low, close, volume
    """
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.sort_index()
    df = df.dropna()
    return df


def generate_ltf_from_htf(htf: pd.DataFrame, ratio: int = 3) -> pd.DataFrame:
    """
    Si tu n'as pas de données LTF, génère une approximation synthétique
    en subdivisant chaque bougie HTF en `ratio` bougies LTF.
    ⚠️  Usage seulement pour tests — utilise de vraies données LTF en production.
    """
    rows = []
    for _, row in htf.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        step = (c - o) / ratio
        for i in range(ratio):
            sub_o = o + step * i
            sub_c = o + step * (i + 1)
            noise = (h - l) * 0.1 * np.random.randn()
            sub_h = max(sub_o, sub_c) + abs(noise)
            sub_l = min(sub_o, sub_c) - abs(noise)
            rows.append({"open": sub_o, "high": sub_h, "low": sub_l, "close": sub_c, "volume": row.get("volume", 0)})
    ltf = pd.DataFrame(rows)
    return ltf


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Découpe temporelle train/validation (jamais aléatoire en trading !)."""
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Callback de log personnalisé
# ──────────────────────────────────────────────────────────────────────────────

class StrategyConformityCallback(BaseCallback):
    """
    Logge régulièrement le taux de conformité à la stratégie
    (% de trades qui avaient un FVG + modèle d'entrée LTF valide).
    """
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.conformity_log = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env.envs[0].env
            stats = env.get_performance_stats()
            conf = stats.get("full_strategy_conformity", 0)
            self.conformity_log.append((self.n_calls, conf))
            if self.verbose:
                print(
                    f"[Step {self.n_calls:>8}] Conformité stratégie : {conf:.1%} "
                    f"| Trades : {stats.get('n_trades', 0)} "
                    f"| Win rate : {stats.get('win_rate', 0):.1%}"
                )
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Entraînement
# ──────────────────────────────────────────────────────────────────────────────

def train(
    htf_path: str,
    ltf_path: Optional[str] = None,
    timesteps: int = 300_000,
    output_dir: str = "models",
    initial_balance: float = 10_000.0,
    risk_pct: float = 0.01,
    seed: int = 42,
    results_dir: str = "results",
):
    if not SB3_AVAILABLE:
        print("Installe stable-baselines3 d'abord : pip install stable-baselines3")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    np.random.seed(seed)
    print(f"🎲 Seed : {seed}")

    print("📂 Chargement des données...")
    htf = load_candles(htf_path)
    
    if ltf_path and os.path.exists(ltf_path):
        ltf = load_candles(ltf_path)
        print(f"✅ HTF : {len(htf)} bougies | LTF : {len(ltf)} bougies")
    else:
        print("⚠️  Pas de données LTF fournies — génération synthétique (ratio x3).")
        ltf = generate_ltf_from_htf(htf, ratio=3)

    # Découpe temporelle
    htf_train, htf_val = split_data(htf, 0.8)
    ratio = max(1, len(ltf) // len(htf))
    n_train_ltf = int(len(htf_train) * ratio)
    ltf_train = ltf.iloc[:n_train_ltf].copy()
    ltf_val   = ltf.iloc[n_train_ltf:].copy()

    print(f"📊 Train : {len(htf_train)} HTF / {len(ltf_train)} LTF")
    print(f"📊 Val   : {len(htf_val)} HTF / {len(ltf_val)} LTF")

    # Création des environnements
    train_env = Monitor(FVGTradingEnv(
        htf_train, ltf_train,
        initial_balance=initial_balance,
        risk_pct=risk_pct,
    ))
    val_env = Monitor(FVGTradingEnv(
        htf_val, ltf_val,
        initial_balance=initial_balance,
        risk_pct=risk_pct,
    ))

    # Vérification de l'env (détecte les bugs)
    print("🔍 Vérification de l'environnement...")
    check_env(train_env, warn=True)
    print("✅ Environnement valide.")

    # ── Agent PPO ─────────────────────────────────────────────────────
    # ent_coef élevé = exploration forcée au début → l'agent ose trader
    # learning_rate réduit = apprentissage plus stable
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,             # horizon plus court → rewards plus immédiats
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,          # augmenté : force l'exploration (trading)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": [256, 128, 64],  # réseau plus large
        },
        verbose=0,
        seed=seed,
        tensorboard_log=f"{results_dir}/tensorboard/",
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=output_dir,
        log_path=results_dir,
        eval_freq=20_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )
    conformity_cb = StrategyConformityCallback(check_freq=20_000, verbose=1)

    # ── Entraînement ──────────────────────────────────────────────────
    print(f"\n🚀 Début de l'entraînement ({timesteps:,} steps)...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, conformity_cb],
        progress_bar=True,
    )

    # ── Sauvegarde ────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "fvg_ppo_final")
    model.save(model_path)
    print(f"\n💾 Modèle sauvegardé : {model_path}.zip")

    # ── Évaluation finale sur le set de validation ────────────────────
    print("\n📈 Évaluation sur le set de validation...")
    stats = evaluate_model(model, htf_val, ltf_val, initial_balance, risk_pct)
    print("\n── Résultats ──────────────────────────────────")
    for k, v in stats.items():
        if k in ("equity_curve", "trades"):
            continue
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")

    # Sauvegarde des stats
    stats_path = os.path.join(results_dir, "eval_stats.json")
    with open(stats_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in stats.items()}, f, indent=2)
    print(f"\n📊 Stats sauvegardées : {stats_path}")

    # Conformité au fil du temps
    if conformity_cb.conformity_log:
        steps, confs = zip(*conformity_cb.conformity_log)
        plt.figure(figsize=(10, 4))
        plt.plot(steps, confs)
        plt.title("Conformité à la stratégie au cours de l'entraînement")
        plt.xlabel("Steps")
        plt.ylabel("Conformité (ratio)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        conformity_path = os.path.join(results_dir, "conformity_curve.png")
        plt.savefig(conformity_path, dpi=150)
        plt.close()
        print(f"📉 Courbe de conformité : {conformity_path}")

    return model, stats


# ──────────────────────────────────────────────────────────────────────────────
# Évaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    htf: pd.DataFrame,
    ltf: pd.DataFrame,
    initial_balance: float = 10_000.0,
    risk_pct: float = 0.01,
) -> dict:
    """Évalue le modèle en mode déterministe sur un dataset complet."""
    env = FVGTradingEnv(htf, ltf, initial_balance=initial_balance, risk_pct=risk_pct)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    stats = env.get_performance_stats()
    stats["equity_curve"] = env.equity_curve
    return stats


def plot_equity_curve(equity_curve: list, output_path: str = "results/equity_curve.png"):
    """Trace et sauvegarde la courbe d'équité."""
    plt.figure(figsize=(12, 5))
    plt.plot(equity_curve, color="#4C9BE8", linewidth=1.5)
    plt.title("Courbe d'équité — Stratégie Base Hits FVG")
    plt.xlabel("Bougies HTF")
    plt.ylabel("Balance ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"📉 Courbe d'équité : {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FVG Base Hits RL agent")
    parser.add_argument("--htf_data",   type=str, required=True,  help="CSV des bougies HTF (15m ou 1h)")
    parser.add_argument("--ltf_data",   type=str, default=None,   help="CSV des bougies LTF (1m ou 5m) — optionnel")
    parser.add_argument("--timesteps",  type=int, default=300_000, help="Nombre de steps d'entraînement")
    parser.add_argument("--balance",    type=float, default=10_000, help="Capital initial")
    parser.add_argument("--risk",       type=float, default=0.01,  help="Risque par trade (ex: 0.01 = 1%%)")
    parser.add_argument("--output_dir", type=str, default="models", help="Dossier de sauvegarde du modèle")
    parser.add_argument("--seed",       type=int, default=42, help="Seed aléatoire (reproductibilité / multi-run)")
    parser.add_argument("--results_dir", type=str, default="results", help="Dossier de sauvegarde des résultats")
    args = parser.parse_args()

    model, stats = train(
        htf_path=args.htf_data,
        ltf_path=args.ltf_data,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        initial_balance=args.balance,
        risk_pct=args.risk,
        seed=args.seed,
        results_dir=args.results_dir,
    )

    if "equity_curve" in stats:
        plot_equity_curve(stats["equity_curve"], output_path=os.path.join(args.results_dir, "equity_curve.png"))