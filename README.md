<div align="center">

# 📈 TNT Prevision - Stock Screener Pro v11

[![CI](https://github.com/nadou25/tnt-prevision/actions/workflows/ci.yml/badge.svg)](https://github.com/nadou25/tnt-prevision/actions/workflows/ci.yml)
[![Daily Scan](https://github.com/nadou25/tnt-prevision/actions/workflows/daily-scan.yml/badge.svg)](https://github.com/nadou25/tnt-prevision/actions/workflows/daily-scan.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🚀 Screener d'actions multi-marchés avec 20+ indicateurs techniques avancés et Machine Learning**

[Fonctionnalités](#-fonctionnalités) •
[Installation](#-installation) •
[Utilisation](#-utilisation) •
[Indicateurs](#-indicateurs-techniques) •
[ML](#-machine-learning)

<img src="https://img.shields.io/badge/Trading-Automatisé-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML-Prédiction-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Multi--Marchés-Global-orange?style=for-the-badge" />

</div>

---

## 🎯 Fonctionnalités

### ⚡ Indicateurs Court Terme
| Indicateur | Description | Signal |
|------------|-------------|--------|
| **ADX + DI+/DI-** | Force de tendance | Trend strength |
| **Stochastic RSI** | Timing optimal | Overbought/Oversold |
| **Williams %R** | Zones extrêmes | Reversal zones |
| **CMF** | Chaikin Money Flow | Buy/Sell pressure |
| **OBV** | On Balance Volume | Accumulation/Distribution |
| **VWAP** | Volume Weighted Price | Fair value |
| **SuperTrend** | Direction tendance | Trend direction |
| **Squeeze Momentum** | Volatilité + Momentum | Breakout detection |

### 🎯 Détection d'Opportunités
- 📊 **Scanner Breakout** - Cassure résistance/support
- 📈 **Détection Gap** - Gap Up/Down significatifs
- 🔥 **Volume Spike Alert** - Volume > 200% moyenne
- 🔄 **Pattern Reversal** - Hammer, Engulfing, Doji
- ⚡ **Momentum Burst** - Accélération soudaine

### 📊 Scoring Multi-Horizon

| Horizon | Durée | Usage | Score |
|---------|-------|-------|-------|
| 🔴 **Intraday** | 1-3 jours | Day trading | 0-100 |
| 🟡 **Swing** | 5-15 jours | Swing trading | 0-100 |
| 🟢 **Position** | 15-60 jours | Position trading | 0-100 |

### 🌍 Marchés Couverts

<div align="center">

| Marché | Couverture | Symboles |
|--------|------------|----------|
| 🇺🇸 **USA** | NYSE, NASDAQ | 500+ |
| 🇪🇺 **Europe** | Euronext, XETRA | 200+ |
| 🇫🇷 **France** | CAC40, SBF120 | 120+ |
| 🎮 **Gaming** | EA, ATVI, UBSFF | 20+ |
| ⛏️ **Commodities** | Or, Pétrole, Gaz | 30+ |
| 🪙 **Crypto** | BTC, ETH, Top 100 | 100+ |

</div>

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- pip

### Installation rapide

```bash
# Cloner le repo
git clone https://github.com/nadou25/tnt-prevision.git
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

---

## 💻 Utilisation

### Lancement

```bash
python stock_screener_pro_v9.py
```

### Options de scan

```bash
# Scan marché US
python stock_screener_pro_v9.py --market us

# Scan crypto
python stock_screener_pro_v9.py --market crypto

# Export Excel
python stock_screener_pro_v9.py --export excel
```

---

## 📊 Indicateurs Techniques

### RSI (Relative Strength Index)
```
RSI < 30  → Survente (BUY signal)
RSI > 70  → Surachat (SELL signal)
```

### MACD
```
MACD > Signal → Bullish
MACD < Signal → Bearish
Histogram ↑   → Momentum croissant
```

### Bollinger Bands
```
Prix < Lower Band → Survente potentielle
Prix > Upper Band → Surachat potentiel
Squeeze          → Breakout imminent
```

---

## 🧠 Machine Learning

### Modèles utilisés

| Modèle | Usage | Accuracy |
|--------|-------|----------|
| **RandomForest** | Classification tendance | ~68% |
| **GradientBoosting** | Scoring opportunités | ~72% |
| **IsolationForest** | Détection anomalies | N/A |
| **AdaBoost** | Ensemble voting | ~70% |

### Features ML
- Prix OHLCV (5, 10, 20, 50 périodes)
- Indicateurs techniques (RSI, MACD, BB, etc.)
- Volume patterns
- Volatilité historique
- Momentum multi-timeframe

---

## 📁 Structure

```
tnt-prevision/
├── stock_screener_pro_v9.py    # 🎯 Application principale
├── requirements.txt            # 📦 Dépendances
├── README.md                   # 📖 Documentation
├── LICENSE                     # 📄 MIT License
├── CHANGELOG.md               # 📝 Historique
├── .github/
│   └── workflows/
│       ├── ci.yml             # ✅ Tests CI
│       └── daily-scan.yml     # 🔄 Scan automatique
└── .gitignore
```

---

## ⚠️ Disclaimer

> **Ce logiciel est fourni à titre éducatif et informatif uniquement.**
> 
> Les signaux générés ne constituent PAS des conseils financiers.
> Le trading comporte des risques significatifs de perte en capital.
> 
> **DYOR - Do Your Own Research**

---

## 📄 License

MIT License - voir [LICENSE](LICENSE)

---

<div align="center">

**Créé avec ❤️ par [Nadir](https://github.com/nadou25)**

[![GitHub](https://img.shields.io/badge/GitHub-nadou25-181717?style=for-the-badge&logo=github)](https://github.com/nadou25)

</div>
