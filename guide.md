# Guide d'utilisation — De zéro à l'agent entraîné

Ce guide couvre toutes les étapes pour installer le projet, télécharger des
données, entraîner l'agent et analyser les résultats. Suis les étapes dans
l'ordre.

---

## Étape 0 — Ce dont tu as besoin

- Un ordinateur sous Windows, macOS ou Linux
- Une connexion internet (pour télécharger Python et les données)
- Environ 500 Mo d'espace disque libre
- Aucune connaissance préalable en Python requise

---

## Étape 1 — Installer Python

### Vérifier si Python est déjà installé

Ouvre un terminal et tape :

```
python --version
```

- **Windows** : touche Windows → tape "cmd" → Entrée
- **macOS** : Cmd + Espace → tape "Terminal" → Entrée
- **Linux** : Ctrl + Alt + T

Si tu vois `Python 3.10.x` ou supérieur, passe à l'étape 2.

### Installer Python si nécessaire

Va sur **https://www.python.org/downloads** et télécharge la dernière version.

> ⚠️ Sur Windows : lors de l'installation, coche obligatoirement
> **"Add Python to PATH"** avant de cliquer sur Install Now.

Vérifie ensuite que l'installation a fonctionné :

```
python --version
```

Tu dois voir un numéro de version s'afficher. Si tu vois une erreur,
redémarre le terminal et réessaie.

---

## Étape 2 — Mettre en place le dossier projet

### Créer le dossier

Crée un dossier nommé `fvg_trader` où tu veux sur ton ordinateur.

Exemple :
- Windows : `C:\Users\TonNom\fvg_trader`
- macOS/Linux : `/home/tonnom/fvg_trader`

### Placer les fichiers

Télécharge tous les fichiers du projet et place-les dans ce dossier.
À la fin tu dois avoir exactement ceci :

```
fvg_trader/
├── strategy.py
├── env.py
├── train.py
├── backtest.py
├── validate.py
├── download_data.py
├── diagnose.py
├── requirements.txt
├── README.md
└── guide.md
```

### Ouvrir un terminal dans ce dossier

- **Windows** : ouvre le dossier dans l'Explorateur → Shift + clic droit
  dans un espace vide → "Ouvrir dans le terminal" (ou "Ouvrir PowerShell ici")
- **macOS** : clic droit sur le dossier dans le Finder → "Nouveau terminal
  au dossier"
- **Linux** : ouvre le terminal, puis tape `cd /chemin/vers/fvg_trader`

Vérifie que tu es bien dans le bon dossier :

```
# Windows
dir

# macOS / Linux
ls
```

Tu dois voir la liste des fichiers du projet s'afficher.

---

## Étape 3 — Créer un environnement virtuel

Un environnement virtuel isole les dépendances du projet pour ne pas
interférer avec d'autres projets Python sur ton ordinateur.

### Créer l'environnement

```
python -m venv venv
```

Cette commande crée un sous-dossier `venv/` dans ton projet.

### Activer l'environnement

Chaque fois que tu ouvres un nouveau terminal, tu dois activer l'environnement
avant de travailler.

```
# Windows (PowerShell)
venv\Scripts\activate

# Windows (CMD)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

Après activation, tu verras `(venv)` apparaître au début de ta ligne de
commande. C'est normal et c'est bon signe.

> ⚠️ Si Windows affiche une erreur "scripts désactivés", tape d'abord :
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> puis réactive l'environnement.

---

## Étape 4 — Installer les dépendances

Avec l'environnement virtuel activé (tu dois voir `(venv)`) :

```
pip install -r requirements.txt
```

L'installation télécharge et installe automatiquement :

- `stable-baselines3[extra]` — l'algorithme PPO pour l'agent RL (inclut tqdm, tensorboard)
- `gymnasium` — le framework d'environnement de simulation
- `numpy` et `pandas` — calcul numérique et manipulation des données
- `matplotlib` — génération des graphiques
- `yfinance` — téléchargement des données historiques

Cela prend entre 1 et 5 minutes selon ta connexion. Tu verras des lignes
défiler. C'est normal.

Une fois terminé, vérifie que tout est bien installé :

```
python -c "import stable_baselines3, gymnasium, pandas, numpy, matplotlib; print('OK')"
```

Si tu vois `OK` s'afficher, tout est en ordre.

---

## Étape 5 — Télécharger les données historiques

### Configuration par défaut

Le fichier `download_data.py` est préconfiguré pour BTC/USD avec les bonnes
unités de temps :

```python
SYMBOL = "BTC-USD"
HTF    = "15m"     # Unité haute : 15 minutes
LTF    = "5m"      # Unité basse : 5 minutes  ← important, pas "1m"
PERIOD = "60d"
```

> ⚠️ **Utilise toujours LTF = "5m"**, pas "1m".
> Yahoo Finance limite le 1m à 7 jours d'historique. Si tu télécharges 60 jours
> de HTF mais seulement 7 jours de LTF, l'agent entraîne sur des données mal
> alignées : les signaux LTF ne correspondent pas aux bonnes bougies HTF.
> Le 5m couvre les 60 jours complets → alignement parfait.

### Changer de marché (optionnel)

Ouvre `download_data.py` avec n'importe quel éditeur de texte et modifie `SYMBOL` :

| Marché | Symbole |
|---|---|
| Bitcoin / Dollar | `"BTC-USD"` |
| Ethereum / Dollar | `"ETH-USD"` |
| Solana / Dollar | `"SOL-USD"` |
| Euro / Dollar | `"EURUSD=X"` |
| Livre / Dollar | `"GBPUSD=X"` |
| Apple | `"AAPL"` |
| Nvidia | `"NVDA"` |

### Lancer le téléchargement

```
python download_data.py
```

Tu dois voir quelque chose comme :

```
══════════════════════════════════════════════════
  FVG Base Hits — Téléchargement des données
══════════════════════════════════════════════════
  Symbole : BTC-USD
  HTF     : 15m  →  data/BTCUSD_15m.csv
  LTF     : 5m   →  data/BTCUSD_5m.csv
  Période : 60d

📡 HTF (15m) :
  ✅ 5760 bougies téléchargées (2024-04-18 → 2024-06-17)
  💾 Sauvegardé : data/BTCUSD_15m.csv

📡 LTF (5m) :
  ✅ 17280 bougies téléchargées (2024-04-18 → 2024-06-17)
  💾 Sauvegardé : data/BTCUSD_5m.csv

  🔗 Chevauchement HTF/LTF : 60.0 jours sur 60.0 jours HTF (100% de couverture)
```

Le **100% de couverture** confirme que les deux fichiers sont bien alignés.
Un score inférieur à 95% signale un problème à régler avant d'entraîner.

---

## Étape 6 — (Optionnel) Diagnostiquer l'environnement

Si l'agent ne trade pas ou si tu veux vérifier que tout fonctionne avant de
lancer un long entraînement :

```
python diagnose.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv
```

Le diagnostic vérifie : la qualité des données, le taux de détection des FVG,
la présence de setups valides, et simule un mini-épisode de 50 bougies.
Tout doit être vert avant de passer à l'étape suivante.

---

## Étape 7 — Entraîner l'agent

### Lancer l'entraînement

```
python train.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv --timesteps 300000
```

### Ce que tu vas voir pendant l'entraînement

```
🎲 Seed : 42
📂 Chargement des données...
✅ HTF : 5760 bougies | LTF : 17280 bougies
📊 Train : 4608 HTF / 13824 LTF
📊 Val   : 1152 HTF / 3456 LTF
🔍 Vérification de l'environnement...
✅ Environnement valide.

🚀 Début de l'entraînement (300 000 steps)...

[Step   20000] Conformité stratégie : 39.0% | Trades : 18 | Win rate : 55.6%
[Step   60000] Conformité stratégie : 72.0% | Trades : 41 | Win rate : 56.1%
[Step  100000] Conformité stratégie : 85.6% | Trades : 63 | Win rate : 57.1%
[Step  200000] Conformité stratégie : 93.7% | Trades : 82 | Win rate : 58.5%
[Step  300000] Conformité stratégie : 96.0% | Trades : 89 | Win rate : 60.7%

💾 Modèle sauvegardé : models/fvg_ppo_final.zip
```

La colonne **Conformité stratégie** est la plus importante : elle mesure le
pourcentage de trades respectant les règles FVG + LTF. Un bon entraînement
doit dépasser **70% dès 60k steps** et se stabiliser au-dessus de **85%**.

### Durée approximative

| CPU | 300 000 steps |
|---|---|
| Ordinateur de bureau moderne | 10 à 20 minutes |
| Laptop milieu de gamme | 20 à 40 minutes |
| Vieux laptop | 40 à 90 minutes |

### Tous les paramètres disponibles

```
--htf_data    Chemin vers le CSV HTF (obligatoire)
--ltf_data    Chemin vers le CSV LTF (optionnel, LTF synthétique si absent)
--timesteps   Nombre de steps (défaut : 300000)
              Recommandé : 150000 pour un test rapide, 300000 pour de bons résultats.
--balance     Capital initial simulé en dollars (défaut : 10000)
--risk        Risque par trade (défaut : 0.01 = 1%)
--output_dir  Dossier de sauvegarde du modèle (défaut : models/)
--seed        Seed aléatoire pour la reproductibilité (défaut : 42)
--results_dir Dossier de sauvegarde des résultats (défaut : results/)
```

Exemples :

```bash
# Entraînement avec une seed précise (reproductible)
python train.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                --timesteps 300000 --seed 123

# Sauvegarder dans des dossiers séparés pour comparer deux runs
python train.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                --timesteps 300000 --seed 42 \
                --output_dir models/run_42 --results_dir results/run_42

# Risque plus élevé, capital plus grand
python train.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                --balance 50000 --risk 0.02 --timesteps 300000
```

---

## Étape 8 — Analyser les résultats

### Lancer le backtest

```
python backtest.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                   --model models/best_model.zip
```

> Le fichier `best_model.zip` est sauvegardé automatiquement quand l'agent
> bat son meilleur score sur la validation. Si tu ne le trouves pas,
> utilise `models/fvg_ppo_final.zip`.

### Ce que le backtest affiche

Une fenêtre s'ouvre avec :

1. **Courbes d'équité** — évolution du capital pour l'agent RL (rouge) vs la
   stratégie pure algorithmique (bleu). Si le rouge est au-dessus, l'agent
   améliore la stratégie.

2. **Graphique de conformité** — win rate, conformité FVG, conformité FVG+LTF.

3. **Tableau de métriques** :

| Métrique | Ce qu'elle mesure |
|---|---|
| PnL total | Profit ou perte en dollars sur la période de test |
| Win rate | % de trades gagnants (>50% est bon pour un ratio 1:1) |
| Max drawdown | Pire perte consécutive en % du capital |
| Sharpe ratio | Rendement ajusté au risque (>1 bien, >2 excellent, >3 remarquable) |
| Conformité FVG | % de trades pris avec un FVG actif |
| Conformité FVG+LTF | % de trades pris avec FVG + Breaker Block confirmé |

Les graphiques sont aussi sauvegardés dans `results/comparison.png`.

---

## Étape 9 — Valider la robustesse avant de faire confiance au modèle

Un bon résultat sur un seul run ne suffit pas — il peut être dû à la chance
sur une seed ou une période de marché particulière. Avant de considérer le
modèle comme fiable, valide sa robustesse avec `validate.py`.

### Pourquoi c'est important

- **Multi-seed** : teste si le comportement est reproductible quelle que soit
  l'initialisation aléatoire du réseau de neurones.
- **Multi-marché** : teste si la stratégie généralise au-delà du seul actif
  d'entraînement.

### Lancer la validation (3 seeds sur le même dataset)

```
python validate.py \
    --datasets BTCUSD:data/BTCUSD_15m.csv:data/BTCUSD_5m.csv \
    --seeds 42 123 2024 \
    --timesteps 150000
```

### Lancer la validation multi-marché (si tu as téléchargé un second actif)

```
python validate.py \
    --datasets BTCUSD:data/BTCUSD_15m.csv:data/BTCUSD_5m.csv \
             ETHUSD:data/ETHUSD_15m.csv:data/ETHUSD_5m.csv \
    --seeds 42 123 \
    --timesteps 150000
```

### Ce que validate.py produit

- Un **tableau comparatif** dans le terminal (PnL, Sharpe, conformité, drawdown par run)
- Un **verdict automatique** : ✅ ROBUSTE / ⚠️ MOYENNEMENT ROBUSTE / ❌ PEU ROBUSTE
- Un **graphique récapitulatif** (`validation_runs/validation_summary.png`)
- Les résultats détaillés de chaque run dans `validation_runs/NOM_seedXXX/`

### Critères pour un modèle fiable

| Indicateur | Seuil minimum | Objectif |
|---|---|---|
| Conformité FVG+LTF | > 70% sur toutes les seeds | > 85% |
| Bat le rule-based | ≥ 2 seeds sur 3 | 3/3 |
| Variance du PnL | CV < 60% | CV < 30% |
| Courbe de conformité | Tendance haussière jusqu'à ~100k steps | Plateau > 80% |

---

## Étape 10 — Réentraîner ou améliorer

### Si la conformité reste sous 70% après 100k steps

Vérifie d'abord que l'alignement HTF/LTF est bon (couverture > 95% dans le
résumé de `download_data.py`). Ensuite, essaie une seed différente — la
progression varie selon l'initialisation.

### Si le modèle est robuste mais le PnL reste proche du rule-based

Augmente les timesteps pour laisser l'agent converger davantage :
```
python train.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                --timesteps 500000 --seed 123
```

### Essayer un autre marché

```
# Modifier SYMBOL dans download_data.py puis :
python download_data.py
python train.py --htf_data data/ETHUSD_15m.csv --ltf_data data/ETHUSD_5m.csv \
                --timesteps 300000
```

---

## Résolution des problèmes courants

### "ModuleNotFoundError: No module named 'stable_baselines3'"

L'environnement virtuel n'est pas activé. Tape :

```
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Puis relance ta commande.

### "python n'est pas reconnu comme une commande"

Python n'est pas dans le PATH. Sur Windows, réinstalle Python en cochant
"Add Python to PATH". Sur macOS, essaie `python3` à la place de `python`.

### "ValueError: Aucune donnée reçue"

Le symbole est incorrect ou Yahoo Finance est temporairement indisponible.
Vérifie sur **finance.yahoo.com** que le symbole est exact.

### Couverture HTF/LTF < 95% dans download_data.py

Tu utilises probablement `LTF = "1m"`. Passe à `LTF = "5m"` dans
`download_data.py` et relance le téléchargement. Le 5m couvre 60 jours,
identique au HTF.

### L'agent reste à 0 trades ou conformité < 20% après 50k steps

Lance d'abord le diagnostic :
```
python diagnose.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv
```
Tout doit être vert. Si la section "Simulation" montre des trades bloqués à
100%, vérifie que tu utilises bien les dernières versions de `strategy.py`
et `env.py`.

### Fenêtre matplotlib qui ne s'ouvre pas

Sur certains serveurs Linux sans interface graphique :
```
export MPLBACKEND=Agg
```
Les graphiques seront sauvegardés dans `results/` sans s'afficher.

---

## Résumé des commandes dans l'ordre

```bash
# 1. Activer l'environnement (à faire à chaque nouveau terminal)
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 2. Télécharger les données (LTF en 5m obligatoire)
python download_data.py

# 3. (Optionnel) Diagnostiquer avant d'entraîner
python diagnose.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv

# 4. Entraîner l'agent
python train.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                --timesteps 300000 --seed 42

# 5. Analyser les résultats
python backtest.py --htf_data data/BTCUSD_15m.csv --ltf_data data/BTCUSD_5m.csv \
                   --model models/best_model.zip

# 6. Valider la robustesse (recommandé avant de faire confiance au modèle)
python validate.py \
    --datasets BTCUSD:data/BTCUSD_15m.csv:data/BTCUSD_5m.csv \
    --seeds 42 123 2024 \
    --timesteps 150000
```