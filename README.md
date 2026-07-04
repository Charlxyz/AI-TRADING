# FVG Base Hits — Agent RL de Trading

Un agent d'intelligence artificielle entraîné à exécuter la stratégie "Base Hits"
basée sur les Fair Value Gaps (FVG). L'IA n'invente pas ses propres règles —
elle apprend à reconnaître les situations où ta stratégie dit d'entrer, et à
prendre la bonne décision au bon moment.

Une fois entraîné, l'agent peut aussi être branché sur un flux de marché en
direct (Binance ou OANDA) pour tourner en **paper trading** — décisions et
PnL réels, mais sans un seul euro/dollar en jeu — sur un actif seul ou sur
plusieurs en parallèle avec un dashboard web.

---

## Le problème que ce projet résout

Appliquer une stratégie de trading à la main, c'est lent, émotionnel, et
impossible à scaler. Un simple algorithme "si condition A alors achète" est
rigide et ne s'adapte pas aux nuances du marché.

Ce projet prend une troisième voie : **entraîner une IA à comprendre et
appliquer ta stratégie** sur des milliers de situations historiques, de façon
à ce qu'elle en capture les subtilités sans jamais dévier de ses règles —
puis observer comment elle se comporte face à un marché réel, en conditions
live, sans risquer de capital.

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

> ⚠️ Utilise le LTF en **5m** (et non 1m) pour l'entraînement à partir de
> Yahoo Finance. Yahoo Finance limite le 1m à 7 jours d'historique, ce qui
> crée un désalignement sévère avec les données HTF sur 60 jours. Le 5m
> couvre les 60 jours complets et assure un apprentissage cohérent. Cette
> contrainte ne s'applique pas au paper trading, qui utilise des flux live
> (Binance/OANDA) sans limite d'historique de ce type.

**Étape 4 — Exécuter avec un ratio risque/rendement de 1:1**
- Stop Loss : serré, placé sur la mèche de la troisième bougie du setup
  (ou sur la distance ATR si elle est plus grande, en paper trading)
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
| Retrade moins de 3 bougies après une clôture | −0.01 (cooldown anti sur-trading) |

L'effet : l'agent comprend rapidement qu'un trade FVG-seul gagnant rapporte
**moins de la moitié** d'un trade conforme équivalent. Il apprend à attendre
la confirmation LTF plutôt que d'agir dès qu'un FVG est présent.

### Ce que l'agent "voit" à chaque instant

L'agent observe un vecteur de 17 valeurs numériques à chaque bougie (+ 4
features supplémentaires sur l'état du portefeuille : position ouverte,
direction, PnL latent, balance normalisée) :

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
├── strategy.py         Détection des FVG, calcul du modèle Breaker Block,
│                       construction des features. Logique pure de la stratégie.
│                       Utilisable indépendamment de l'agent RL.
│
├── env.py              Environnement de simulation (compatible Gymnasium).
│                       Gère les trades, le SL/TP automatique, le reward shaping.
│
├── train.py             Script d'entraînement. Charge les données, sépare
│                       train/validation, entraîne l'agent PPO, sauvegarde le
│                       meilleur modèle. Supporte --seed et --results_dir pour
│                       des runs reproductibles et isolés.
│
├── backtest.py          Compare l'agent RL contre la stratégie pure
│                       algorithmique (rule-based) sur les données de test.
│                       Génère les graphiques de performance.
│
├── validate.py          Validation de robustesse multi-seed / multi-marché.
│                       Lance plusieurs runs en séquence, backteste chacun,
│                       et produit un rapport comparatif automatique.
│
├── download_data.py     Télécharge les données HTF et LTF depuis Yahoo Finance
│                       (crypto, Forex, actions). Configuré par défaut en
│                       HTF=15m / LTF=5m pour un alignement complet sur 60j.
│
├── diagnose.py          Diagnostic complet de l'environnement et des données.
│                       À lancer si l'agent ne trade pas ou si les FVG ne
│                       sont pas détectés.
│
├── live_fetcher.py      Récupération des bougies et du prix en temps réel
│                       (API publique Binance, ou OANDA pour Forex/Or/Indices).
│                       Fournit `LiveDataFeed` (un seul actif) et
│                       `MarketDataHub` (mutualisation multi-actifs).
│
├── paper_engine.py      Moteur de paper trading : construit l'observation,
│                       interroge l'agent PPO, vérifie la conformité FVG+LTF,
│                       ouvre/ferme les trades simulés, calcule les stats,
│                       persiste l'historique en JSON.
│
├── paper_trading.py     Script principal du paper trading MONO-ACTIF.
│                       Se connecte à Binance ou OANDA, pilote un dashboard
│                       terminal (curses) en temps réel, ou un mode console.
│
├── dashboard.py         Dashboard terminal (curses) affichant en direct :
│                       position ouverte, signal courant, conformité,
│                       métriques (win rate, Sharpe, drawdown), mini courbe
│                       d'équité ASCII et historique des trades.
│
├── paper_report.py      Génère un rapport HTML statique et autonome à partir
│                       d'un fichier paper_trades.json (courbe d'équité,
│                       cartes de métriques, tableau des trades).
│
├── server.py             Serveur MULTI-ACTIFS. Fait tourner N combinaisons
│                       (symbole, HTF, LTF) en parallèle à partir de
│                       config.json, mutualise les appels réseau via
│                       MarketDataHub, et diffuse l'état de chaque actif en
│                       temps réel via WebSocket vers un dashboard web
│                       (dashboard.html, servi en HTTP sur le même process).
│
├── config.json           Configuration du mode multi-actifs : capital,
│                       risque, levier et intervalles HTF/LTF par actif,
│                       activable/désactivable individuellement.
│
├── requirements.txt      Dépendances Python.
│
├── data/                 Dossier pour les CSV de données historiques.
├── models/               Dossier où les modèles entraînés sont sauvegardés.
├── results/              Dossier pour les graphiques et statistiques de sortie.
├── validation_runs/      Dossier créé par validate.py (résultats détaillés
│                       de chaque run de validation).
└── paper_sessions/       Dossier créé par paper_trading.py / server.py,
                        un sous-dossier par symbole, contenant l'historique
                        JSON de chaque session de paper trading.
```

---

## Paper trading — tester l'agent en conditions réelles, sans risque

Une fois un modèle entraîné et validé, l'étape suivante consiste à observer
comment il se comporte face à un marché qui bouge réellement, avec des
données qu'il n'a jamais vues (ni à l'entraînement, ni en backtest). C'est le
rôle du paper trading : **aucun ordre réel n'est envoyé**, mais toutes les
décisions, tous les prix d'entrée/sortie et tout le PnL sont calculés comme
si l'argent était réel.

### Mode mono-actif — `paper_trading.py`

```
python paper_trading.py --model models/best_model.zip --symbol BTCUSDT
```

Ce script :
1. Charge le modèle PPO entraîné
2. Se connecte à Binance (API publique, sans clé) ou à OANDA (Forex/Or/Indices,
   avec `--oanda_key` et `--oanda_account`)
3. Attend la fermeture de chaque bougie HTF
4. Demande une décision à l'agent (HOLD / BUY / SELL)
5. Vérifie la conformité stratégie (FVG actif + modèle d'entrée LTF) et
   **bloque physiquement** l'ouverture d'un trade sans FVG actif, comme à
   l'entraînement
6. Exécute ou refuse le trade en papier, gère le SL/TP automatique
7. Affiche un dashboard terminal en temps réel (`dashboard.py`), ou bascule
   en mode console avec `--no_dashboard`
8. Sauvegarde l'historique dans `paper_sessions/SYMBOL/SYMBOL_htf_ltf.json`

Le dashboard terminal affiche : prix et balance en direct, la position
ouverte (direction, SL, TP, PnL latent), le dernier signal de l'agent avec
son statut de conformité, les métriques de performance cumulées, une mini
courbe d'équité ASCII et l'historique des 10 derniers trades. Touches :
`q` pour quitter, `c` pour clôturer manuellement une position.

Options utiles : `--balance`, `--risk`, `--leverage` (x1 par défaut),
`--htf`/`--ltf` (défaut 15m/5m).

### Rapport HTML — `paper_report.py`

```
python paper_report.py --symbol BTCUSDT
```

Génère un rapport HTML autonome (thème sombre) à partir du JSON de session :
cartes de métriques (balance, PnL, win rate, drawdown, Sharpe, conformité),
courbe d'équité, et tableau complet des trades avec badges de statut
(✅ TP / ❌ SL / ✋ manuel) et de conformité stratégie.

### Mode multi-actifs — `server.py`

Pour faire tourner plusieurs combinaisons (actif, HTF, LTF) en parallèle avec
un seul modèle, sans multiplier les appels réseau :

```
python server.py --model models/best_model.zip --config config.json
```

- Chaque entrée de `config.json` (voir plus bas) devient un `SymbolWorker`
  indépendant : sa propre balance, son propre risque, son propre levier,
  son propre historique.
- Un seul `MarketDataHub` mutualise les requêtes Binance : si deux workers
  utilisent le même (symbole, intervalle) — par ex. BTCUSDT 15m pour les
  combos 15m/1m *et* 15m/5m — la donnée n'est téléchargée qu'une seule fois.
- Un thread unique (`MarketPoller`) rafraîchit le hub puis déclenche chaque
  worker.
- L'état de tous les actifs est diffusé une fois par seconde via WebSocket
  (`ws://localhost:PORT+1/`) vers `dashboard.html`, servi en HTTP sur
  `http://localhost:PORT/`.
- Commande supportée depuis le dashboard web : clôture manuelle d'une
  position (`close_position`).

### Format de `config.json`

```json
{
  "global": {
    "initial_balance": 1000.0,
    "risk_pct": 0.01,
    "leverage": 1,
    "refresh_interval": 5,
    "web_port": 8765
  },
  "symbols": [
    {"symbol": "BTCUSDT", "htf": "15m", "ltf": "5m", "balance": 1000.0, "risk_pct": 0.01, "leverage": 1, "enabled": true}
  ]
}
```

Chaque entrée de `symbols` peut surcharger `balance`, `risk_pct` et
`leverage` individuellement ; `enabled: false` désactive un actif sans le
supprimer de la config.

> ⚠️ Le levier (`leverage`) amplifie le PnL (gains ET pertes) proportionnellement
> à la marge immobilisée — il n'est jamais activé par défaut (x1) et doit être
> augmenté en connaissance de cause, même en paper trading, pour que les
> chiffres observés restent représentatifs d'un usage réel.

---

## Ce que le projet ne fait pas

- Il n'envoie **aucun ordre réel** à un broker ou une bourse — le paper
  trading calcule tout en simulation, y compris en mode multi-actifs
- Il ne cherche pas à maximiser le profit à tout prix — il cherche à maximiser
  la conformité à ta stratégie ET le profit dans ce cadre
- Il ne garantit pas des résultats futurs (aucun backtest, ni aucune session
  de paper trading, ne peut le garantir)

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
| Données historiques | yfinance / CSV |
| Données live | API publique Binance, API OANDA (Forex/Or/Indices) |
| Calcul numérique | NumPy, Pandas |
| Visualisation (backtest/validation) | Matplotlib |
| Dashboard terminal (paper trading) | curses |
| Dashboard web multi-actifs | WebSocket (`websockets`), HTML/JS (dashboard.html) |
| Rapport de session | HTML autonome (paper_report.py) |