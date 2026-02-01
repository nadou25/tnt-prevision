# 📈 TNT Prevision - Stock Screener Pro v11

> Screener d'actions multi-marchés avec 20+ indicateurs techniques avancés et Machine Learning.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vue d'ensemble

**TNT Prevision** est un screener d'actions professionnel qui combine analyse technique avancée et Machine Learning pour détecter les meilleures opportunités court et moyen terme.

### ⚡ Indicateurs Court Terme
- **ADX + DI+/DI-** - Force de tendance
- **Stochastic RSI** - Timing optimal
- **Williams %R** - Zones extrêmes
- **CMF** - Chaikin Money Flow (pression achat/vente)
- **OBV** - On Balance Volume (accumulation/distribution)
- **VWAP** - Volume Weighted Average Price
- **SuperTrend** - Direction de tendance
- **Squeeze Momentum** - Volatilité + Momentum

### 🎯 Détection d'Opportunités
- Scanner Breakout (cassure résistance/support)
- Détection Gap (Gap Up/Down significatifs)
- Volume Spike Alert (volume > 200% moyenne)
- Pattern Reversal (Hammer, Engulfing, Doji)
- Momentum Burst (accélération soudaine)

### 📊 Scoring Multi-Horizon
| Horizon | Durée | Usage |
|---------|-------|-------|
| Intraday | 1-3 jours | Day trading |
| Swing | 5-15 jours | Swing trading |
| Position | 15-60 jours | Position trading |

### 🌍 Marchés Couverts
- 🇺🇸 USA (NYSE, NASDAQ)
- 🇪🇺 Europe (Euronext, XETRA)
- 🇫🇷 France (CAC40, SBF120)
- 🎮 Gaming (EA, Activision, Ubisoft...)
- ⛏️ Commodities (Or, Pétrole, Gaz...)
- 🪙 Crypto (Top 100)

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- pip

### Installation rapide

```bash
# Cloner le repo
git clone https://github.com/YOUR_USERNAME/tnt-prevision.git
cd tnt-prevision

# Créer environnement virtuel
python -m venv .venv

# Activer (Windows)
.venv\Scripts\activate

# Activer (Linux/Mac)
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement

```bash
python stock_screener_pro_v9.py
```

---

## 📁 Structure

```
tnt-prevision/
├── stock_screener_pro_v9.py    # Application principale
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation
├── LICENSE                     # Licence MIT
└── .gitignore                  # Fichiers ignorés
```

---

## 🔧 Configuration

### Variables d'environnement (optionnel)

```bash
# Fichier .env
LOG_LEVEL=INFO
CACHE_TTL=3600
```

---

## 📊 Utilisation

### Mode interactif

```bash
python stock_screener_pro_v9.py
```

### Options disponibles
- Scanner les marchés US
- Scanner les marchés européens
- Scanner les cryptos
- Exporter en Excel/CSV

---

## 🧠 Machine Learning

Le screener utilise plusieurs modèles ML pour améliorer les prédictions :

| Modèle | Usage |
|--------|-------|
| RandomForest | Classification tendance |
| GradientBoosting | Scoring opportunités |
| IsolationForest | Détection anomalies |
| AdaBoost | Ensemble voting |

---

## ⚠️ Disclaimer

> **Ce logiciel est fourni à titre éducatif et informatif uniquement.**
> 
> Les signaux générés ne constituent PAS des conseils financiers professionnels.
> Le trading comporte des risques significatifs de perte en capital.
> 
> **Faites toujours vos propres recherches (DYOR) avant d'investir.**

---

## 📄 License

MIT License - voir [LICENSE](LICENSE)

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md)

---

Créé avec ❤️ par **Nadir**
