"""
validate.py — Valide la robustesse de l'agent FVG Base Hits

Lance plusieurs entraînements (croisement seeds x datasets), backteste
chacun sur son propre set de validation, et produit un tableau comparatif
pour répondre à deux questions :

  1. Est-ce que le bon résultat tient sur plusieurs seeds (même données) ?
     → si la conformité et le PnL varient énormément d'une seed à l'autre,
       le run initial était en grande partie de la chance.

  2. Est-ce que ça marche aussi sur d'autres marchés/périodes ?
     → si les résultats s'effondrent sur un autre actif, le modèle a
       probablement appris des particularités de ce marché précis plutôt
       que la stratégie elle-même.

Usage basique (3 seeds sur le même dataset BTC) :
    python validate.py \
        --datasets BTCUSD:data/BTCUSD_15m.csv:data/BTCUSD_5m.csv \
        --seeds 42 123 2024 \
        --timesteps 150000

Usage multi-marché (en plus, un second actif) :
    python validate.py \
        --datasets BTCUSD:data/BTCUSD_15m.csv:data/BTCUSD_5m.csv ETHUSD:data/ETHUSD_15m.csv:data/ETHUSD_5m.csv \
        --seeds 42 123 \
        --timesteps 150000

⚠️ timesteps plus bas que 300k par défaut : avec plusieurs runs à enchaîner,
   150k est un compromis raisonnable. Augmente si tu as le temps.
"""

import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import train, load_candles, generate_ltf_from_htf, split_data
from backtest import run_pure_strategy, run_rl_agent


def parse_dataset_arg(s: str):
    """Parse 'NAME:htf_path:ltf_path' ou 'NAME:htf_path' (sans LTF)."""
    parts = s.split(":")
    if len(parts) == 3:
        return {"name": parts[0], "htf": parts[1], "ltf": parts[2]}
    elif len(parts) == 2:
        return {"name": parts[0], "htf": parts[1], "ltf": None}
    raise ValueError(f"Format dataset invalide : '{s}' (attendu NAME:htf.csv:ltf.csv)")


def run_one(dataset: dict, seed: int, timesteps: int, base_dir: str,
            initial_balance: float = 10_000.0, risk_pct: float = 0.01) -> dict:
    """Entraîne + backteste un agent sur un (dataset, seed) donné."""
    run_id = f"{dataset['name']}_seed{seed}"
    run_dir = os.path.join(base_dir, run_id)
    output_dir = os.path.join(run_dir, "models")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"  RUN : {run_id}  (timesteps={timesteps:,})")
    print("═" * 70)

    model, train_stats = train(
        htf_path=dataset["htf"],
        ltf_path=dataset["ltf"],
        timesteps=timesteps,
        output_dir=output_dir,
        initial_balance=initial_balance,
        risk_pct=risk_pct,
        seed=seed,
        results_dir=results_dir,
    )

    # ── Backtest out-of-sample (mêmes 20% que train.py a utilisés en val) ──
    htf = load_candles(dataset["htf"])
    if dataset["ltf"] and os.path.exists(dataset["ltf"]):
        ltf = load_candles(dataset["ltf"])
    else:
        ltf = generate_ltf_from_htf(htf, ratio=3)

    htf_train, htf_val = split_data(htf, 0.8)
    ratio = max(1, len(ltf) // len(htf))
    n_train_ltf = int(len(htf_train) * ratio)
    ltf_val = ltf.iloc[n_train_ltf:].copy()

    stats_rb = run_pure_strategy(htf_val, ltf_val, initial_balance, risk_pct)

    best_model_path = os.path.join(output_dir, "best_model.zip")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(output_dir, "fvg_ppo_final.zip")
    stats_rl = run_rl_agent(best_model_path, htf_val, ltf_val, initial_balance, risk_pct)

    # Conformité finale d'entraînement (dernier point loggé)
    conformity_log_path = os.path.join(results_dir, "eval_stats.json")
    final_conformity = stats_rl.get("full_strategy_conformity", 0.0)

    result = {
        "run_id": run_id,
        "dataset": dataset["name"],
        "seed": seed,
        "rb_pnl": stats_rb.get("total_pnl", 0.0),
        "rb_sharpe": stats_rb.get("sharpe_ratio", 0.0),
        "rb_winrate": stats_rb.get("win_rate", 0.0),
        "rb_trades": stats_rb.get("n_trades", 0),
        "rl_pnl": stats_rl.get("total_pnl", 0.0),
        "rl_sharpe": stats_rl.get("sharpe_ratio", 0.0),
        "rl_winrate": stats_rl.get("win_rate", 0.0),
        "rl_trades": stats_rl.get("n_trades", 0),
        "rl_conformity": final_conformity,
        "rl_drawdown": stats_rl.get("max_drawdown", 0.0),
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_report(results: list):
    """Affiche un tableau comparatif et un verdict de robustesse."""
    print("\n" + "═" * 100)
    print("  RAPPORT DE VALIDATION DE ROBUSTESSE")
    print("═" * 100)

    header = f"{'Run':<22}{'PnL RL':>10}{'Sharpe RL':>11}{'WR RL':>8}{'Conform.':>10}{'DD RL':>8}{'PnL RB':>10}{'Sharpe RB':>11}"
    print(header)
    print("─" * 100)
    for r in results:
        print(
            f"{r['run_id']:<22}"
            f"{r['rl_pnl']:>+10.2f}"
            f"{r['rl_sharpe']:>11.3f}"
            f"{r['rl_winrate']:>8.1%}"
            f"{r['rl_conformity']:>10.1%}"
            f"{r['rl_drawdown']:>8.1%}"
            f"{r['rb_pnl']:>+10.2f}"
            f"{r['rb_sharpe']:>11.3f}"
        )

    print("─" * 100)

    # ── Analyse multi-seed (même dataset) ──────────────────────────────
    by_dataset = {}
    for r in results:
        by_dataset.setdefault(r["dataset"], []).append(r)

    print("\n  ── Stabilité par dataset (sur les seeds testées) ──\n")
    for name, runs in by_dataset.items():
        if len(runs) < 2:
            print(f"  {name} : une seule seed testée, pas d'analyse de variance possible.")
            continue
        pnls = np.array([r["rl_pnl"] for r in runs])
        sharpes = np.array([r["rl_sharpe"] for r in runs])
        conf = np.array([r["rl_conformity"] for r in runs])
        beats_rb = np.array([r["rl_pnl"] > r["rb_pnl"] for r in runs])

        print(f"  {name} ({len(runs)} seeds) :")
        print(f"    PnL RL        : moyenne={pnls.mean():+.2f}  écart-type={pnls.std():.2f}  min={pnls.min():+.2f}  max={pnls.max():+.2f}")
        print(f"    Sharpe RL     : moyenne={sharpes.mean():.3f}  écart-type={sharpes.std():.3f}")
        print(f"    Conformité RL : moyenne={conf.mean():.1%}  min={conf.min():.1%}  max={conf.max():.1%}")
        print(f"    Bat le rule-based : {beats_rb.sum()}/{len(runs)} seeds")

        # Verdict simple basé sur le coefficient de variation du PnL
        cv = abs(pnls.std() / pnls.mean()) if pnls.mean() != 0 else float("inf")
        if cv < 0.3 and beats_rb.mean() >= 0.66:
            verdict = "✅ ROBUSTE — résultats cohérents sur les seeds testées."
        elif cv < 0.6:
            verdict = "⚠️  MOYENNEMENT ROBUSTE — variance notable, à confirmer avec plus de seeds/steps."
        else:
            verdict = "❌ PEU ROBUSTE — forte variance, le run initial était probablement en partie de la chance."
        print(f"    Verdict : {verdict}")
        print()

    # ── Analyse multi-marché (entre datasets) ──────────────────────────
    if len(by_dataset) > 1:
        print("  ── Généralisation entre marchés/périodes ──\n")
        for name, runs in by_dataset.items():
            avg_conf = np.mean([r["rl_conformity"] for r in runs])
            avg_pnl = np.mean([r["rl_pnl"] for r in runs])
            print(f"  {name} : conformité moyenne={avg_conf:.1%}  PnL moyen={avg_pnl:+.2f}")
        confs = [np.mean([r["rl_conformity"] for r in runs]) for runs in by_dataset.values()]
        if max(confs) - min(confs) > 0.25:
            print("\n  ⚠️  Écart important de conformité entre marchés : le modèle généralise mal,")
            print("     il a probablement surappris des particularités d'un marché précis.")
        else:
            print("\n  ✅ Conformité cohérente entre marchés testés — bon signe de généralisation.")

    print("\n" + "═" * 100)


def plot_summary(results: list, output_path: str = "results/validation_summary.png"):
    """Graphique récapitulatif : PnL et conformité par run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [r["run_id"] for r in results]
    pnls = [r["rl_pnl"] for r in results]
    rb_pnls = [r["rb_pnl"] for r in results]
    confs = [r["rl_conformity"] for r in results]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, rb_pnls, width, label="Rule-based", color="#5B8CDB", alpha=0.85)
    axes[0].bar(x + width/2, pnls, width, label="RL Agent", color="#E87A5D", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("PnL par run")
    axes[0].set_ylabel("PnL ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25, axis="y")
    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(x, confs, color="#6FBF73", alpha=0.85)
    axes[1].axhline(0.7, color="red", linestyle="--", linewidth=1, label="Seuil 'bon' (70%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Conformité FVG+LTF par run")
    axes[1].set_ylabel("Conformité")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Graphique récapitulatif : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation de robustesse multi-seed / multi-marché")
    parser.add_argument(
        "--datasets", type=str, nargs="+", required=True,
        help="Liste de 'NAME:htf.csv:ltf.csv' (ou 'NAME:htf.csv' sans LTF)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--base_dir", type=str, default="validation_runs")
    args = parser.parse_args()

    datasets = [parse_dataset_arg(d) for d in args.datasets]

    print(f"📋 Plan de validation : {len(datasets)} dataset(s) × {len(args.seeds)} seed(s) "
          f"= {len(datasets) * len(args.seeds)} runs, {args.timesteps:,} steps chacun.")

    os.makedirs(args.base_dir, exist_ok=True)
    all_results = []
    for dataset in datasets:
        for seed in args.seeds:
            try:
                result = run_one(
                    dataset, seed, args.timesteps, args.base_dir,
                    initial_balance=args.balance, risk_pct=args.risk,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n❌ Échec sur {dataset['name']} / seed={seed} : {e}")

    if not all_results:
        print("\n❌ Aucun run n'a abouti. Vérifie les chemins de données.")
    else:
        print_report(all_results)
        plot_summary(all_results, os.path.join(args.base_dir, "validation_summary.png"))

        with open(os.path.join(args.base_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"📁 Résultats complets : {os.path.join(args.base_dir, 'all_results.json')}")
