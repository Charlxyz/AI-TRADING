"""
download_data.py — Téléchargement des données historiques pour la stratégie Base Hits FVG

Usage :
    python download_data.py

Télécharge automatiquement :
  - HTF (15m) → data/BTCUSD_15m.csv
  - LTF (1m)  → data/BTCUSD_1m.csv

Modifie les variables en haut du fichier pour changer le symbole ou les dates.
"""

import os
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIGURATION — modifie ces variables selon tes besoins
# ──────────────────────────────────────────────────────────────────────────────

SYMBOL   = "BTC-USD"     # Symbole yfinance. Exemples :
                          #   Crypto  : "BTC-USD", "ETH-USD", "SOL-USD"
                          #   Forex   : "EURUSD=X", "GBPUSD=X", "USDJPY=X"
                          #   Actions : "AAPL", "TSLA", "NVDA"
                          #   Indices : "^GSPC" (S&P500), "^NDX" (Nasdaq)

HTF      = "15m"         # Unité de temps haute  : "15m" ou "1h"
LTF      = "5m"          # Unité de temps basse  : "1m" ou "5m"
                          # ⚠️ "1m" est limité à 7j par Yahoo Finance, ce qui ne
                          # couvre qu'une fraction du HTF (60j) → désalignement
                          # massif lors de l'entraînement. "5m" couvre 60j,
                          # comme le HTF : utilise "5m" sauf besoin spécifique.

# Période à télécharger (limites yfinance) :
#   "1m"  → max 7 jours
#   "5m"  → max 60 jours
#   "15m" → max 60 jours
#   "1h"  → max 730 jours (2 ans)
PERIOD   = "60d"         # Pour HTF 1h : "60d" est le maximum

OUTPUT_DIR = "data"

# ──────────────────────────────────────────────────────────────────────────────
# Téléchargement
# ──────────────────────────────────────────────────────────────────────────────

def install_yfinance():
    """Installe yfinance si pas encore présent."""
    try:
        import yfinance
    except ImportError:
        print("📦 Installation de yfinance...")
        os.system("pip install yfinance --quiet")


def clean_name(symbol: str) -> str:
    """Transforme 'BTC-USD' en 'BTCUSD', 'EURUSD=X' en 'EURUSD', etc."""
    return symbol.replace("-", "").replace("=X", "").replace("^", "")


def download(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Télécharge et nettoie les données OHLCV."""
    import yfinance as yf

    print(f"  ⏳ Téléchargement {symbol} [{interval}] sur {period}...")
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"Aucune donnée reçue pour {symbol} [{interval}]. "
            "Vérifie le symbole ou la période."
        )

    # Aplatit les colonnes multi-index si nécessaire (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalise les noms de colonnes
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})

    # Garde uniquement les colonnes utiles
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols]

    # Supprime les bougies avec des valeurs manquantes ou nulles
    df = df.dropna()
    df = df[df["close"] > 0]

    # Trie par date croissante
    df = df.sort_index()

    print(f"  ✅ {len(df)} bougies téléchargées "
          f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")

    return df


def save(df: pd.DataFrame, path: str):
    """Sauvegarde le DataFrame en CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True, index_label="datetime")
    size_kb = os.path.getsize(path) / 1024
    print(f"  💾 Sauvegardé : {path} ({size_kb:.1f} Ko)")


def check_alignment(htf: pd.DataFrame, ltf: pd.DataFrame):
    """
    Vérifie que les deux datasets couvrent la même période, avec un vrai
    pourcentage de couverture (pas juste un chevauchement binaire).
    Un faible pourcentage signifie qu'une grande partie de l'entraînement
    couplera du HTF avec un LTF non représentatif de la même période.
    """
    htf_start = htf.index[0]
    htf_end   = htf.index[-1]
    ltf_start = ltf.index[0]
    ltf_end   = ltf.index[-1]

    overlap_start = max(htf_start, ltf_start)
    overlap_end   = min(htf_end, ltf_end)

    htf_span_days = max((htf_end - htf_start).total_seconds() / 86400, 1e-9)

    if overlap_start >= overlap_end:
        print("  ⚠️  ATTENTION : les datasets HTF et LTF ne se chevauchent PAS du tout !")
        print("  → L'entraînement avec ces deux fichiers n'a pas de sens (le LTF")
        print("    sera systématiquement désynchronisé du HTF).")
        return 0.0

    overlap_days = (overlap_end - overlap_start).total_seconds() / 86400
    coverage_pct = min(overlap_days / htf_span_days * 100, 100.0)

    print(f"  🔗 Chevauchement HTF/LTF : {overlap_days:.1f} jours sur {htf_span_days:.1f} jours HTF "
          f"({coverage_pct:.0f}% de couverture)")

    if coverage_pct < 50:
        print(f"  ⚠️  ATTENTION : seulement {coverage_pct:.0f}% du HTF est couvert par le LTF.")
        print("     Plus de la moitié de l'entraînement va coupler du HTF avec un LTF")
        print("     hors période → signaux d'entrée LTF non significatifs sur cette portion.")
        print("     Conseil : utilise LTF='5m' (couvre 60j, comme le HTF en 15m/1h).")
    elif coverage_pct < 95:
        print(f"  ℹ️  Couverture partielle ({coverage_pct:.0f}%) — acceptable mais pas idéale.")

    return coverage_pct


def print_summary(htf: pd.DataFrame, ltf: pd.DataFrame, name: str):
    """Affiche un résumé des données téléchargées."""
    print()
    print("─" * 50)
    print(f"  Résumé — {name}")
    print("─" * 50)
    print(f"  HTF ({HTF}) : {len(htf):>6} bougies")
    print(f"  LTF ({LTF}) : {len(ltf):>6} bougies")
    print(f"  Ratio LTF/HTF        : {len(ltf) / len(htf):.1f}x")
    print(f"  Prix actuel (clôture): {htf['close'].iloc[-1]:.4f}")
    print(f"  Plus haut période    : {htf['high'].max():.4f}")
    print(f"  Plus bas période     : {htf['low'].min():.4f}")
    volatilite = htf['close'].pct_change().std() * 100
    print(f"  Volatilité (std %)   : {volatilite:.3f}%")
    print("─" * 50)
    print()
    print("  Prochaine étape :")
    htf_file = os.path.join(OUTPUT_DIR, f"{name}_{HTF}_aligned.csv")
    if not os.path.exists(htf_file):
        htf_file = os.path.join(OUTPUT_DIR, f"{name}_{HTF}.csv")
    ltf_file = os.path.join(OUTPUT_DIR, f"{name}_{LTF}.csv")
    print(f"  python train.py --htf_data {htf_file} \\")
    print(f"                  --ltf_data {ltf_file} \\")
    print(f"                  --timesteps 300000")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    install_yfinance()

    name = clean_name(SYMBOL)
    htf_path = os.path.join(OUTPUT_DIR, f"{name}_{HTF}.csv")
    ltf_path = os.path.join(OUTPUT_DIR, f"{name}_{LTF}.csv")

    print()
    print("═" * 50)
    print("  FVG Base Hits — Téléchargement des données")
    print("═" * 50)
    print(f"  Symbole : {SYMBOL}")
    print(f"  HTF     : {HTF}  →  {htf_path}")
    print(f"  LTF     : {LTF}  →  {ltf_path}")
    print(f"  Période : {PERIOD}")
    print()

    # Téléchargement HTF
    print(f"📡 HTF ({HTF}) :")
    htf = download(SYMBOL, HTF, PERIOD)
    save(htf, htf_path)

    # Téléchargement LTF
    # La période LTF est limitée à 7j pour le 1m — on prend le max possible
    ltf_period = "7d" if LTF == "1m" else "60d"
    print(f"\n📡 LTF ({LTF}) :")
    if ltf_period != PERIOD:
        print(f"  ℹ️  Période réduite à {ltf_period} (limite yfinance pour {LTF})")
    ltf = download(SYMBOL, LTF, ltf_period)
    save(ltf, ltf_path)

    # ── Alignement : on réduit le HTF pour couvrir la même période que le LTF ──
    # C'est crucial : si HTF couvre 60j et LTF seulement 7j,
    # l'entraînement n'a pas de données LTF pour 53j → l'agent ne peut rien apprendre.
    if len(ltf) > 0 and ltf_period != PERIOD:
        ltf_start = ltf.index[0]
        htf_aligned = htf[htf.index >= ltf_start].copy()
        if len(htf_aligned) > 50:
            print(f"\n  🔧 HTF aligné sur la période LTF : {len(htf_aligned)} bougies conservées")
            htf_aligned_path = htf_path.replace(f"_{HTF}.csv", f"_{HTF}_aligned.csv")
            save(htf_aligned, htf_aligned_path)
            print(f"  ℹ️  Utilise ce fichier pour l'entraînement : {htf_aligned_path}")
            htf = htf_aligned
            htf_path = htf_aligned_path
        else:
            print(f"\n  ⚠️  Pas assez de données HTF dans la période LTF ({len(htf_aligned)} bougies).")
            print(f"  → Conseil : passe LTF = '5m' dans la config (période 60j disponible).")

    # Vérification de l'alignement
    print()
    check_alignment(htf, ltf)

    # Résumé final
    print_summary(htf, ltf, name)