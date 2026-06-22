"""
strategy.py — Logique pure de la stratégie "Base Hits" (FVG)
Détecte les FVG sur HTF et les signaux d'entrée sur LTF.
Aucune dépendance à l'agent RL — utilisable en backtest classique aussi.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class FVG:
    """Représente un Fair Value Gap détecté."""
    direction: str          # 'bullish' ou 'bearish'
    top: float              # borne haute du gap
    bottom: float           # borne basse du gap
    candle_idx: int         # index de la 3e bougie (celle qui crée le gap)
    confirmed: bool = False # la 3e bougie respecte la condition de confirmation


def detect_fvg(candles: pd.DataFrame, idx: int) -> Optional[FVG]:
    """
    Détecte un FVG sur les 3 bougies se terminant à idx.

    Un FVG bullish : mèche haute de la bougie 1 < mèche basse de la bougie 3
      → zone vide entre high[i-2] et low[i]
    Un FVG bearish : mèche basse de la bougie 1 > mèche haute de la bougie 3
      → zone vide entre low[i-2] et high[i]

    Condition de confirmation (étape 2 de la stratégie) :
      - FVG bearish (signal SELL) : bougie 3 ne clôture PAS au-dessus du high de la bougie 2
      - FVG bullish (signal BUY)  : bougie 3 ne clôture PAS en dessous du low de la bougie 2
    """
    if idx < 2:
        return None

    c1 = candles.iloc[idx - 2]
    c2 = candles.iloc[idx - 1]
    c3 = candles.iloc[idx]

    # FVG bearish : les vendeurs ont créé un vide → opportunité BUY
    if c1["low"] > c3["high"]:
        gap_top = c1["low"]
        gap_bottom = c3["high"]
        confirmed = c3["close"] >= c2["low"]  # ne clôture pas sous le low de c2
        return FVG("bearish", gap_top, gap_bottom, idx, confirmed)

    # FVG bullish : les acheteurs ont créé un vide → opportunité SELL
    if c1["high"] < c3["low"]:
        gap_top = c3["low"]
        gap_bottom = c1["high"]
        confirmed = c3["close"] <= c2["high"]  # ne clôture pas au-dessus du high de c2
        return FVG("bullish", gap_top, gap_bottom, idx, confirmed)

    return None


def scan_active_fvgs(candles: pd.DataFrame, current_idx: int, lookback: int = 20) -> list[FVG]:
    """
    Retourne les FVG encore "actifs" (non comblés) dans la fenêtre récente.
    Un FVG est comblé quand le prix retraverse ENTIEREMENT la zone du gap
    depuis l'exterieur (pas seulement "se trouve" d'un cote d'une borne,
    ce qui est trivialement vrai des la bougie suivant la creation).
    """
    active = []
    start = max(2, current_idx - lookback)
    # current_idx inclus : un FVG forme sur la bougie courante doit pouvoir
    # etre vu comme actif au meme pas de temps (sinon il n'est jamais detecte
    # avant d'etre potentiellement deja comble).
    for i in range(start, current_idx + 1):
        fvg = detect_fvg(candles, i)
        if fvg is None or not fvg.confirmed:
            continue

        # Le comblement ne peut etre verifie qu'APRES la bougie de creation (i).
        # On exige une traversee complete de la zone (entree par le bon cote
        # ET sortie par l'autre bord), pas un simple chevauchement d'une meche.
        filled = False
        for j in range(i + 1, min(current_idx, len(candles) - 1) + 1):
            c = candles.iloc[j]
            if fvg.direction == "bearish":
                # Gap au-dessus du prix (signal BUY). Comble quand le prix
                # revient depuis le bas et traverse toute la zone jusqu'au top.
                if c["high"] >= fvg.top:
                    filled = True
                    break
            else:  # bullish
                # Gap en dessous du prix (signal SELL). Comble quand le prix
                # redescend depuis le haut et traverse toute la zone jusqu'au bottom.
                if c["low"] <= fvg.bottom:
                    filled = True
                    break

        if not filled:
            active.append(fvg)

    return active


def detect_entry_model_ltf(ltf_candles: pd.DataFrame, idx: int, trade_dir: str) -> float:
    """
    Détecte un modèle d'entrée sur LTF (Breaker Block simplifié).
    Retourne un score entre 0 et 1 (0 = pas de signal, 1 = signal fort).

    Breaker Block BUY  : Bas → Haut → Plus Bas → casse le Haut précédent
    Breaker Block SELL : Haut → Bas → Plus Haut → casse le Bas précédent

    On simplifie ici : on cherche une structure de retournement sur 4 bougies.
    """
    if idx < 3:
        return 0.0

    c1 = ltf_candles.iloc[idx - 3]
    c2 = ltf_candles.iloc[idx - 2]
    c3 = ltf_candles.iloc[idx - 1]
    c4 = ltf_candles.iloc[idx]

    score = 0.0

    if trade_dir == "buy":
        # Séquence : bas, haut, plus bas, puis casse du haut
        swing_low_1  = c1["low"]
        swing_high   = c2["high"]
        swing_low_2  = c3["low"]
        current_high = c4["high"]

        if swing_low_2 < swing_low_1 and current_high > swing_high:
            score = 1.0
        elif swing_low_2 < swing_low_1:
            score = 0.5  # structure partielle

    elif trade_dir == "sell":
        swing_high_1 = c1["high"]
        swing_low    = c2["low"]
        swing_high_2 = c3["high"]
        current_low  = c4["low"]

        if swing_high_2 > swing_high_1 and current_low < swing_low:
            score = 1.0
        elif swing_high_2 > swing_high_1:
            score = 0.5

    return score


def compute_state_features(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_idx: int,
    ltf_idx: int,
) -> np.ndarray:
    """
    Construit le vecteur d'état pour l'agent RL.
    Combine les données HTF (contexte FVG) et LTF (modèle d'entrée).

    Features (17 dimensions) :
      [0]  Prix normalisé (close / close_ref - 1)
      [1]  Variation HTF (close-to-close)
      [2]  FVG bearish actif ? (0/1)
      [3]  Distance au top du FVG bearish (normalisée)
      [4]  Distance au bottom du FVG bearish (normalisée)
      [5]  FVG bullish actif ? (0/1)
      [6]  Distance au top du FVG bullish (normalisée)
      [7]  Distance au bottom du FVG bullish (normalisée)
      [8]  Score breaker block BUY sur LTF
      [9]  Score breaker block SELL sur LTF
      [10] Momentum LTF court (3 bougies)
      [11] Volatilité LTF (ATR normalisé sur 5 bougies)
      [12] Spread high-low de la dernière bougie LTF normalisé
      [13] Close HTF vs open HTF (direction de la bougie)
      [14] Nombre de FVG actifs (normalisé)
      [15] Confirmation de la 3e bougie HTF (0/1)
      [16] Position dans la session (0→1, heure normalisée si dispo)
    """
    features = np.zeros(17, dtype=np.float32)

    if htf_idx < 2 or ltf_idx < 3:
        return features

    htf = htf_candles.iloc[htf_idx]
    ref_price = htf_candles.iloc[htf_idx - 1]["close"]

    # [0] Prix normalisé
    features[0] = (htf["close"] / ref_price) - 1.0

    # [1] Variation HTF
    features[1] = (htf["close"] - htf["open"]) / htf["open"]

    # FVG actifs
    active_fvgs = scan_active_fvgs(htf_candles, htf_idx, lookback=10)
    bearish_fvgs = [f for f in active_fvgs if f.direction == "bearish"]
    bullish_fvgs = [f for f in active_fvgs if f.direction == "bullish"]
    current_price = htf["close"]

    # [2-4] FVG bearish le plus proche (signal BUY)
    if bearish_fvgs:
        closest = min(bearish_fvgs, key=lambda f: abs(current_price - (f.top + f.bottom) / 2))
        features[2] = 1.0
        features[3] = (closest.top - current_price) / current_price
        features[4] = (closest.bottom - current_price) / current_price
    
    # [5-7] FVG bullish le plus proche (signal SELL)
    if bullish_fvgs:
        closest = min(bullish_fvgs, key=lambda f: abs(current_price - (f.top + f.bottom) / 2))
        features[5] = 1.0
        features[6] = (closest.top - current_price) / current_price
        features[7] = (closest.bottom - current_price) / current_price

    # [8-9] Modèles d'entrée LTF
    features[8]  = detect_entry_model_ltf(ltf_candles, ltf_idx, "buy")
    features[9]  = detect_entry_model_ltf(ltf_candles, ltf_idx, "sell")

    # [10] Momentum LTF court
    if ltf_idx >= 3:
        ltf_close = ltf_candles["close"]
        features[10] = (ltf_close.iloc[ltf_idx] - ltf_close.iloc[ltf_idx - 3]) / ltf_close.iloc[ltf_idx - 3]

    # [11] ATR LTF normalisé
    if ltf_idx >= 5:
        window = ltf_candles.iloc[ltf_idx - 5: ltf_idx + 1]
        atr = (window["high"] - window["low"]).mean()
        features[11] = atr / current_price

    # [12] Spread dernière bougie LTF
    ltf_last = ltf_candles.iloc[ltf_idx]
    features[12] = (ltf_last["high"] - ltf_last["low"]) / ltf_last["close"]

    # [13] Direction bougie HTF
    features[13] = 1.0 if htf["close"] > htf["open"] else -1.0

    # [14] Nombre de FVG actifs
    features[14] = min(len(active_fvgs) / 5.0, 1.0)

    # [15] Confirmation 3e bougie HTF
    fvg_now = detect_fvg(htf_candles, htf_idx)
    features[15] = 1.0 if (fvg_now and fvg_now.confirmed) else 0.0

    # [16] Heure normalisée (si index datetime)
    try:
        hour = htf_candles.index[htf_idx].hour
        features[16] = hour / 23.0
    except Exception:
        features[16] = 0.5

    return np.clip(features, -5.0, 5.0)