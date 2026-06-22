"""
diagnose.py — Diagnostic complet avant entraînement

Vérifie :
  1. Les données HTF/LTF (format, taille, valeurs)
  2. La détection de FVG (combien se forment, sont-ils confirmés ?)
  3. Les modèles d'entrée LTF (Breaker Block)
  4. Le vecteur d'état (features non nulles ?)
  5. Un épisode manuel pour voir ce que l'agent "voit"

Usage :
    python diagnose.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_1m.csv
"""

import argparse
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from strategy import detect_fvg, scan_active_fvgs, detect_entry_model_ltf, compute_state_features


def load(path):
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.lower().strip() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df.sort_index().dropna()


def section(title):
    print()
    print("═" * 55)
    print(f"  {title}")
    print("═" * 55)


def check(label, ok, detail=""):
    symbol = "✅" if ok else "❌"
    print(f"  {symbol}  {label}", f"→ {detail}" if detail else "")


# ─────────────────────────────────────────────────────────
# 1. Données
# ─────────────────────────────────────────────────────────

def diagnose_data(htf, ltf):
    section("1. DONNÉES")

    check("HTF chargé", len(htf) > 0, f"{len(htf)} bougies")
    check("LTF chargé", len(ltf) > 0, f"{len(ltf)} bougies")
    check("HTF assez long", len(htf) >= 100, f"min recommandé : 100")
    check("LTF assez long", len(ltf) >= 300, f"min recommandé : 300")

    # Colonnes
    for col in ["open", "high", "low", "close"]:
        check(f"Colonne HTF '{col}'", col in htf.columns)
        check(f"Colonne LTF '{col}'", col in ltf.columns)

    # Valeurs aberrantes
    htf_zeros = (htf[["open","high","low","close"]] == 0).sum().sum()
    check("Pas de prix à 0 dans HTF", htf_zeros == 0, f"{htf_zeros} valeurs nulles")

    ltf_zeros = (ltf[["open","high","low","close"]] == 0).sum().sum()
    check("Pas de prix à 0 dans LTF", ltf_zeros == 0, f"{ltf_zeros} valeurs nulles")

    # Cohérence high >= low
    htf_bad = (htf["high"] < htf["low"]).sum()
    check("HTF : high >= low", htf_bad == 0, f"{htf_bad} bougies incohérentes")

    ltf_bad = (ltf["high"] < ltf["low"]).sum()
    check("LTF : high >= low", ltf_bad == 0, f"{ltf_bad} bougies incohérentes")

    # Chevauchement temporel
    try:
        htf_start, htf_end = htf.index[0], htf.index[-1]
        ltf_start, ltf_end = ltf.index[0], ltf.index[-1]
        overlap = min(htf_end, ltf_end) > max(htf_start, ltf_start)
        check("HTF et LTF se chevauchent", overlap,
              f"HTF: {htf_start.date()} → {htf_end.date()} | "
              f"LTF: {ltf_start.date()} → {ltf_end.date()}")
        if overlap:
            days = (min(htf_end, ltf_end) - max(htf_start, ltf_start)).days
            check("Chevauchement suffisant", days >= 3, f"{days} jours en commun")
    except Exception:
        print("  ⚠️  Index non-datetime, comparaison temporelle ignorée")

    print(f"\n  HTF prix moyen : {htf['close'].mean():.4f}")
    print(f"  LTF prix moyen : {ltf['close'].mean():.4f}")
    print(f"  HTF volatilité (std) : {htf['close'].pct_change().std()*100:.4f}%")

    ratio = len(ltf) / max(len(htf), 1)
    check("Ratio LTF/HTF raisonnable", 0.5 <= ratio <= 20,
          f"{ratio:.1f}x (attendu ~3x pour 15m/5m ou ~15x pour 15m/1m)")


# ─────────────────────────────────────────────────────────
# 2. Détection FVG
# ─────────────────────────────────────────────────────────

def diagnose_fvg(htf):
    section("2. DÉTECTION DES FVG SUR HTF")

    all_fvgs = []
    for i in range(2, len(htf)):
        fvg = detect_fvg(htf, i)
        if fvg:
            all_fvgs.append(fvg)

    total = len(all_fvgs)
    confirmed = [f for f in all_fvgs if f.confirmed]
    bearish = [f for f in all_fvgs if f.direction == "bearish"]
    bullish = [f for f in all_fvgs if f.direction == "bullish"]

    check("Au moins 1 FVG détecté", total > 0, f"{total} FVG sur {len(htf)} bougies")
    check("Taux de FVG > 1%", total / len(htf) > 0.01,
          f"{total/len(htf)*100:.1f}% des bougies génèrent un FVG")
    check("FVG bearish présents", len(bearish) > 0, f"{len(bearish)}")
    check("FVG bullish présents", len(bullish) > 0, f"{len(bullish)}")
    check("FVG confirmés", len(confirmed) > 0,
          f"{len(confirmed)}/{total} confirmés ({len(confirmed)/max(total,1)*100:.0f}%)")

    if total == 0:
        print()
        print("  ⚠️  PROBLÈME CRITIQUE : aucun FVG détecté.")
        print("  Causes possibles :")
        print("  - Les bougies sont trop petites (marché peu volatile)")
        print("  - Les colonnes high/low sont identiques ou proches de 0")
        print("  - Les données ne correspondent pas à la résolution attendue")
        print()
        print("  Aperçu des 5 premières bougies HTF :")
        print(htf[["open","high","low","close"]].head().to_string())
        return

    if total > 0:
        gaps = [f.top - f.bottom for f in all_fvgs]
        print(f"\n  Taille moyenne du FVG : {np.mean(gaps):.6f}")
        print(f"  Taille médiane du FVG : {np.median(gaps):.6f}")
        print(f"  Plus grand FVG        : {max(gaps):.6f}")

    # Test scan_active_fvgs au milieu du dataset
    mid = len(htf) // 2
    active = scan_active_fvgs(htf, mid, lookback=20)
    print(f"\n  FVG actifs (non comblés) à la bougie {mid} : {len(active)}")
    for f in active:
        print(f"    → {f.direction:8s} | top={f.top:.4f} bottom={f.bottom:.4f} "
              f"confirmé={f.confirmed}")

    active_end = scan_active_fvgs(htf, len(htf)-1, lookback=20)
    check("FVG actifs en fin de dataset", len(active_end) > 0,
          f"{len(active_end)} FVG non comblés dans les 20 dernières bougies")


# ─────────────────────────────────────────────────────────
# 3. Modèles d'entrée LTF
# ─────────────────────────────────────────────────────────

def diagnose_ltf(ltf):
    section("3. MODÈLES D'ENTRÉE LTF (BREAKER BLOCK)")

    buy_signals = 0
    sell_signals = 0
    partial_buy = 0
    partial_sell = 0

    for i in range(3, len(ltf)):
        b = detect_entry_model_ltf(ltf, i, "buy")
        s = detect_entry_model_ltf(ltf, i, "sell")
        if b >= 1.0: buy_signals += 1
        elif b >= 0.5: partial_buy += 1
        if s >= 1.0: sell_signals += 1
        elif s >= 0.5: partial_sell += 1

    total = len(ltf) - 3
    check("Signaux BUY complets détectés", buy_signals > 0,
          f"{buy_signals} ({buy_signals/total*100:.1f}% des bougies)")
    check("Signaux SELL complets détectés", sell_signals > 0,
          f"{sell_signals} ({sell_signals/total*100:.1f}% des bougies)")
    check("Signaux partiels BUY", partial_buy > 0, f"{partial_buy}")
    check("Signaux partiels SELL", partial_sell > 0, f"{partial_sell}")

    if buy_signals == 0 and sell_signals == 0:
        print()
        print("  ⚠️  Aucun modèle d'entrée LTF détecté.")
        print("  Causes possibles :")
        print("  - LTF trop court (moins de 10 bougies)")
        print("  - Données LTF incohérentes")
        print()
        print("  Aperçu LTF :")
        print(ltf[["open","high","low","close"]].head(10).to_string())


# ─────────────────────────────────────────────────────────
# 4. Features (vecteur d'état)
# ─────────────────────────────────────────────────────────

def diagnose_features(htf, ltf):
    section("4. VECTEUR D'ÉTAT (FEATURES)")

    # On teste plusieurs points dans le dataset
    test_indices = [10, len(htf)//4, len(htf)//2, len(htf)*3//4, len(htf)-5]
    ltf_ratio = max(1, len(ltf) // len(htf))

    non_zero_counts = []
    fvg_feature_counts = []

    for htf_idx in test_indices:
        ltf_idx = min(htf_idx * ltf_ratio, len(ltf) - 1)
        feats = compute_state_features(htf, ltf, htf_idx, ltf_idx)
        non_zero = (feats != 0).sum()
        non_zero_counts.append(non_zero)
        # features [2] et [5] = FVG bearish/bullish actif
        fvg_active = feats[2] + feats[5]
        fvg_feature_counts.append(fvg_active)

    avg_nonzero = np.mean(non_zero_counts)
    avg_fvg = np.mean(fvg_feature_counts)

    check("Features non-nulles en moyenne", avg_nonzero > 3,
          f"{avg_nonzero:.1f}/17 features non nulles en moyenne")
    check("Feature FVG active au moins parfois", avg_fvg > 0,
          f"FVG actif en moyenne {avg_fvg:.2f}/2.0 aux points de test")

    print(f"\n  Détail par point de test :")
    feature_names = [
        "prix normalisé", "variation HTF", "FVG bearish actif",
        "dist top bearish", "dist bot bearish", "FVG bullish actif",
        "dist top bullish", "dist bot bullish", "score BUY LTF",
        "score SELL LTF", "momentum LTF", "ATR LTF", "spread LTF",
        "direction HTF", "nb FVG actifs", "confirmation c3", "heure"
    ]

    # Affiche les features au milieu du dataset
    mid = len(htf) // 2
    ltf_mid = min(mid * ltf_ratio, len(ltf) - 1)
    feats_mid = compute_state_features(htf, ltf, mid, ltf_mid)

    print(f"\n  Features à la bougie {mid} :")
    for i, (name, val) in enumerate(zip(feature_names, feats_mid)):
        flag = " ◄ FVG!" if i in [2, 5] and val > 0 else ""
        flag = " ◄ SIGNAL!" if i in [8, 9] and val > 0 else flag
        print(f"    [{i:02d}] {name:<22} = {val:+.6f}{flag}")

    if avg_fvg == 0:
        print()
        print("  ⚠️  Les features FVG sont toujours à 0 aux points de test.")
        print("  → L'agent ne voit jamais de FVG : il ne peut pas apprendre")
        print("    à les respecter car ils n'existent pas dans son observation.")


# ─────────────────────────────────────────────────────────
# 5. Simulation d'un mini-épisode
# ─────────────────────────────────────────────────────────

def diagnose_episode(htf, ltf):
    section("5. SIMULATION D'UN MINI-ÉPISODE (50 bougies)")

    from env import FVGTradingEnv

    env = FVGTradingEnv(htf, ltf, initial_balance=10_000.0)
    obs, _ = env.reset()

    fvg_seen = 0
    actions_taken = {0: 0, 1: 0, 2: 0}
    rewards = []
    blocked = 0

    for step in range(min(50, len(htf) - 10)):
        # Simule les 3 actions pour voir les rewards disponibles
        htf_idx = env.htf_step
        ltf_idx = env.ltf_step
        current_price = float(env.htf.iloc[htf_idx]["close"])

        active = scan_active_fvgs(env.htf, htf_idx, lookback=10)
        if active:
            fvg_seen += 1

        # Prend l'action "acheter" pour tester
        if env.open_trade is None:
            obs, reward, term, trunc, _ = env.step(1)
            rewards.append(reward)
            actions_taken[1] += 1
            if reward == -0.03:  # blocage dur
                blocked += 1
        else:
            obs, reward, term, trunc, _ = env.step(0)
            rewards.append(reward)
            actions_taken[0] += 1

        if term or trunc:
            break

    check("FVG vus pendant l'épisode", fvg_seen > 0,
          f"{fvg_seen}/50 bougies avaient un FVG actif")
    check("Trades non bloqués", blocked < 50,
          f"{blocked}/50 tentatives bloquées (pas de FVG)")
    check("Rewards variés (pas tous identiques)", len(set(rewards)) > 2,
          f"{len(set(rewards))} valeurs uniques de reward")

    if rewards:
        print(f"\n  Reward moyen   : {np.mean(rewards):+.4f}")
        print(f"  Reward min     : {min(rewards):+.4f}")
        print(f"  Reward max     : {max(rewards):+.4f}")
        print(f"  Trades bloqués : {blocked}/50 ({blocked/50*100:.0f}%)")

    if blocked == 50:
        print()
        print("  ⚠️  PROBLÈME CONFIRMÉ : 100% des trades sont bloqués.")
        print("  L'agent ne peut physiquement jamais ouvrir un trade")
        print("  car aucun FVG n'est détecté sur ces données.")
        print()
        print("  SOLUTIONS :")
        print("  1. Utilise les données non-alignées (plus de bougies = plus de FVG)")
        print("     → python train.py --htf_data data/BTCUSD_15m.csv (sans ltf_data)")
        print()
        print("  2. Augmente le lookback FVG dans strategy.py :")
        print("     → scan_active_fvgs(..., lookback=50) au lieu de 10")
        print()
        print("  3. Assouplis les conditions FVG (voir ci-dessous)")


# ─────────────────────────────────────────────────────────
# 6. Résumé et recommandations
# ─────────────────────────────────────────────────────────

def summary(htf, ltf):
    section("6. RÉSUMÉ ET COMMANDE RECOMMANDÉE")

    ltf_ratio = max(1, len(ltf) // len(htf))

    # Compte les FVG actifs sur tout le dataset
    fvg_count = 0
    for i in range(10, len(htf)):
        active = scan_active_fvgs(htf, i, lookback=20)
        if active:
            fvg_count += 1

    fvg_pct = fvg_count / (len(htf) - 10) * 100

    print(f"  Bougies HTF avec FVG actif : {fvg_count}/{len(htf)-10} ({fvg_pct:.1f}%)")
    print()

    if fvg_pct < 5:
        print("  ⚠️  Moins de 5% des bougies ont un FVG actif.")
        print("  L'agent apprendra difficilement — peu de situations d'entraînement.")
        print()
        print("  → Recommandation : passe à HTF=1h avec plus de données")
        print("    Le 1h génère moins de bougies mais des FVG plus nets et plus durables.")
        print()
        print("  Dans download_data.py, change :")
        print("    HTF = '1h'")
        print("    LTF = '5m'")
        print("    PERIOD = '730d'  ← 2 ans de données 1h")
        print()
        print("  Puis relance :")
        print("    python download_data.py")
        print("    python train.py --htf_data data/BTCUSD_1h.csv --ltf_data data/BTCUSD_5m.csv --timesteps 500000")
    elif fvg_pct < 15:
        print("  ✅ Taux de FVG correct. L'entraînement devrait fonctionner.")
        print("     Si la conformité reste à 0%, augmente les timesteps.")
    else:
        print("  ✅ Bon taux de FVG. L'agent a suffisamment de situations à apprendre.")

    print()
    print("  Commande d'entraînement recommandée :")
    print(f"  python train.py \\")
    print(f"    --htf_data data/BTCUSD_15m.csv \\")
    print(f"    --ltf_data data/BTCUSD_1m.csv \\")
    print(f"    --timesteps 500000")
    print()


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--htf_data", required=True)
    parser.add_argument("--ltf_data", default=None)
    args = parser.parse_args()

    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║     FVG Base Hits — Diagnostic de l'environnement    ║")
    print("╚═══════════════════════════════════════════════════════╝")

    htf = load(args.htf_data)
    ltf = load(args.ltf_data) if args.ltf_data else htf.copy()

    diagnose_data(htf, ltf)
    diagnose_fvg(htf)
    diagnose_ltf(ltf)
    diagnose_features(htf, ltf)
    diagnose_episode(htf, ltf)
    summary(htf, ltf)