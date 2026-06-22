# FVG Base Hits — Agent RL de Trading

Un agent d'intelligence artificielle entraîné à exécuter la stratégie "Base Hits"
basée sur les Fair Value Gaps (FVG). L'IA n'invente pas ses propres règles —
elle apprend à reconnaître les situations où ta stratégie dit d'entrer, et à
prendre la bonne décision au bon moment.

---

## Le problème que ce projet résout

Appliquer une stratégie de trading à la main, c'est lent, émotionnel, et
impossible à scaler. Un simple algorithme "si condition A alors achète" est
rigide et ne s'adapte pas aux nuances du marché.

Ce projet prend une troisième voie : **entraîner une IA à comprendre et
appliquer ta stratégie** sur des milliers de situations historiques, de façon
à ce qu'elle en capture les subtilités sans jamais dévier de ses règles.

---

## La stratégie "Base Hits" — en résumé

La stratégie repose sur une idée simple : quand le marché se déplace très
rapidement dans une direction, il laisse des **zones vides** (Fair Value Gaps)
qu'il reviendra souvent combler. L'objectif est d'anticiper ce retour.

### Ce qu'est un Fair Value Gap (FVG)

Un FVG est une zone de prix où les échanges ont été si rapides qu'il n'y a
eu pratiquement aucune transaction. Il se repère sur trois bougies
consécutives : la mèche de la première bougie et la mèche de la troisième
ne se touchent pas. L'espace entre les deux est le FVG.

```
Signal BUY (FVG bearish)       Signal SELL (FVG bullish)

  ─┐                               ─┐  ← mèche haute c1
   │  ← c1                          │
  ─┘                               ─┘
    ← ZONE VIDE (FVG)            ─┐  ← mèche haute c3
  ─┐                               │  ← c3
   │  ← c3                         │
  ─┘                              ─┘
                                   ← ZONE VIDE (FVG)
Le prix doit revenir              ─┐  ← c1
combler vers le haut.              └─
```

### Les 4 étapes de la stratégie

**Étape 1 — Identifier le FVG sur le graphique HTF (15m ou 1h)**
On attend qu'un FVG se forme. Un FVG bearish (vide créé par les vendeurs)
au-dessus du prix actuel donne un signal BUY potentiel. Un FVG bullish
(vide créé par les acheteurs) en dessous donne un signal SELL potentiel.

**Étape 2 — Confirmer la 3e bougie**
La troisième bougie doit respecter une condition précise pour valider le
setup. Pour un BUY : elle ne doit pas clôturer sous le plus bas de la
deuxième bougie. Pour un SELL : elle ne doit pas clôturer au-dessus du
plus haut de la deuxième bougie. Cette confirmation prouve que des
acheteurs ou vendeurs réactifs sont bien présents.

**Étape 3 — Trouver le modèle d'entrée sur LTF (5m)**
Une fois le FVG validé sur le HTF, on descend sur une unité de temps
inférieure pour chercher un signal d'entrée précis. Le modèle principal
utilisé est le **Breaker Block** : une structure de retournement à quatre
bougies qui confirme le changement de direction.

> ⚠️ Utilise le LTF en **5m** (et non 1m). Yahoo Finance limite le 1m à
> 7 jours d'historique, ce qui crée un désalignement sévère avec les données
> HTF sur 60 jours. Le 5m couvre les 60 jours complets et assure un
> apprentissage cohérent.

**Étape 4 — Exécuter avec un ratio risque/rendement de 1:1**
- Stop Loss : serré, placé sur la mèche de la troisième bougie du setup
- Take Profit : équivalent au risque (ratio 1:1), visant le comblement du FVG

---

## Comment l'IA apprend cette stratégie

### L'approche : Reinforcement Learning (RL)

L'agent utilise l'algorithme **PPO** (Proximal Policy Optimization), une
méthode d'apprentissage par renforcement. Concrètement :

1. L'agent observe l'état du marché (prix, FVG actifs, signaux LTF)
2. Il choisit une action : acheter, vendre, ou ne rien faire
3. Il reçoit une récompense (reward) selon que son action était bonne
4. Il répète cela des centaines de milliers de fois sur les données historiques
5. À force de répétition, il apprend quand agir et quand attendre

### Ce qui force l'IA à respecter ta stratégie

Le cœur du système est le **reward shaping** : la façon dont on définit ce
qui est "bien" ou "mal" pour l'agent.

| Situation | Récompense |
|---|---|
| Trade ouvert sans FVG actif | −0.03 (pénalité forte, trade physiquement refusé) |
| Trade ouvert avec FVG mais sans confirmation LTF | −0.015 (pénalité à l'entrée) |
| Trade ouvert avec FVG + modèle LTF | +0.08 (bonus immédiat fort) |
| Win conforme (FVG + LTF) | Reward × 1.5 (gains fortement valorisés) |
| Win FVG seul (sans LTF) | Reward × 0.6 (gains nettement écrêtés) |
| Loss hors-stratégie (FVG sans LTF) | Reward × 1.15 (perte légèrement amplifiée) |
| Tentative d'ouvrir un 2e trade simultané | −0.005 |

L'effet : l'agent comprend rapidement qu'un trade FVG-seul gagnant rapporte
**moins de la moitié** d'un trade conforme équivalent. Il apprend à attendre
la confirmation LTF plutôt que d'agir dès qu'un FVG est présent.

### Ce que l'agent "voit" à chaque instant

L'agent observe un vecteur de 17 valeurs numériques à chaque bougie :

- Prix normalisé et variation HTF récente
- Présence de FVG bearish/bullish actifs et leur distance au prix actuel
- Score du modèle d'entrée LTF (Breaker Block) pour BUY et pour SELL
- Momentum court terme sur le LTF
- Volatilité récente (ATR normalisé)
- Spread de la dernière bougie LTF
- Direction de la bougie HTF courante
- Nombre de FVG actifs
- Confirmation de la 3e bougie HTF
- Position dans la session (heure normalisée)

---

## Structure du projet

```
fvg_trader/
│
├── strategy.py        Détection des FVG, calcul du modèle Breaker Block,
│                      construction des features. Logique pure de la stratégie.
│                      Utilisable indépendamment de l'agent RL.
│
├── env.py             Environnement de simulation (compatible OpenAI Gym).
│                      Gère les trades, le SL/TP automatique, le reward shaping.
│
├── train.py           Script d'entraînement. Charge les données, sépare
│                      train/validation, entraîne l'agent PPO, sauvegarde le
│                      meilleur modèle. Supporte --seed et --results_dir pour
│                      des runs reproductibles et isolés.
│
├── backtest.py        Compare l'agent RL contre la stratégie pure
│                      algorithmique (rule-based) sur les données de test.
│                      Génère les graphiques de performance.
│
├── validate.py        Validation de robustesse multi-seed / multi-marché.
│                      Lance plusieurs runs en séquence, backteste chacun,
│                      et produit un rapport comparatif automatique pour
│                      confirmer que les résultats ne sont pas de la chance.
│
├── download_data.py   Télécharge les données HTF et LTF depuis Yahoo Finance
│                      (crypto, Forex, actions). Configuré par défaut en
│                      HTF=15m / LTF=5m pour un alignement complet sur 60j.
│
├── diagnose.py        Diagnostic complet de l'environnement et des données.
│                      À lancer si l'agent ne trade pas ou si les FVG ne
│                      sont pas détectés.
│
├── requirements.txt   Dépendances Python.
│
├── data/              Dossier pour les CSV de données historiques.
├── models/            Dossier où les modèles entraînés sont sauvegardés.
├── results/           Dossier pour les graphiques et statistiques de sortie.
└── validation_runs/   Dossier créé par validate.py, contenant les résultats
                       détaillés de chaque run de validation (modèles, stats,
                       courbes de conformité).
```

---

## Ce que le projet ne fait pas

- Il ne trade pas en live (pas de connexion à un broker)
- Il ne cherche pas à maximiser le profit à tout prix — il cherche à maximiser
  la conformité à ta stratégie ET le profit dans ce cadre
- Il ne garantit pas des résultats futurs (aucun backtest ne le peut)

---

## Résultats obtenus (BTC/USD 15m, validation multi-seed)

Après validation sur 3 seeds indépendantes (42, 123, 2024) avec 150k steps chacune :

| Seed | PnL RL | Sharpe RL | Conformité FVG+LTF | Bat le rule-based |
|---|---|---|---|---|
| 42 | +1 450$ | — | 87% | ✅ |
| 123 | +2 050$ | — | 100% | ✅ |
| 2024 | +1 580$ | — | 95% | ✅ |

Les 3 seeds battent le rule-based (+1 340$) avec une conformité minimum de 87%.
Le pattern d'apprentissage est reproductible et cohérent entre initialisations.

---

## Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Agent RL | PPO — stable-baselines3 |
| Environnement | OpenAI Gymnasium |
| Données | yfinance / CSV |
| Calcul numérique | NumPy, Pandas |
| Visualisation | Matplotlib |