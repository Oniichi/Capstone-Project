# Gold Price Forecasting: ML/DL Model Comparison

**A comprehensive capstone project comparing baseline statistical models, machine learning, and deep learning approaches for predicting short-term gold price movements.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📊 Quick Results

Evaluated **7 models** across **9 years** (2016-2024) using walk-forward validation:

| Model | RMSE ↓ | MAE ↓ | Dir. Accuracy ↑ |
|-------|--------|-------|-----------------|
| **Random Forest** | **0.00890** | **0.00640** | **50.81%** |
| SMA(20) | 0.00891 | 0.00643 | 50.46% |
| XGBoost | 0.00929 | 0.00687 | 47.48% |
| LSTM | 0.00937 | 0.00691 | 49.66% |
| GRU | 0.00960 | 0.00709 | 51.11% |
| Naive | 0.01249 | 0.00945 | 44.97% |
| MLP | 0.02699 | 0.02302 | 49.70% |

**🏆 Winner:** Random Forest (best RMSE)  
**📌 Key Finding:** Simple baselines are competitive—gold prices exhibit mean-reversion patterns.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Everything Automatically
```bash
python scripts/run_all_notebooks.py
```

This single command:
- Executes all 8 notebooks in order
- Trains all 7 models
- Generates metrics and visualizations
- Creates final summary report

**Estimated runtime:** ~30-45 minutes

### 3. View Results
```
results/metrics/walk_forward_results.csv    # Performance metrics
results/figures/                             # Generated plots
docs/RESULTS.md                              # Analysis summary
```

---

## 📚 Documentation

| Document | Content | Read Time |
|----------|---------|-----------|
| **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** | Data sources, features, models, validation strategy | 10 min |
| **[docs/RESULTS.md](docs/RESULTS.md)** | Detailed metrics, visualizations, key findings | 10 min |
| **[docs/CONCLUSIONS.md](docs/CONCLUSIONS.md)** | Insights, recommendations, limitations | 5 min |

---

## 📁 Project Structure

```
gold-capstone-project/
├── README.md                          # ← You are here
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT License
├── .gitignore
│
├── docs/                              # 📖 Documentation
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   ├── CONCLUSIONS.md
│   └── figures/                       # All visualizations
│
├── notebooks/                         # 📓 Analysis (8 notebooks)
│   ├── README.md                      # Notebook guide
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_ml_models.ipynb
│   ├── 07_deep_learning.ipynb
│   └── 08_walk_forward_validation.ipynb
│
├── scripts/                           # 🚀 Automation
│   └── run_all_notebooks.py
│
├── data/                              # 📊 Data (gitignored)
│   ├── raw/                           # Original sources
│   └── processed/                     # Cleaned, engineered
│
└── results/                           # 📈 Outputs (gitignored)
    ├── metrics/
    └── figures/
```

---

## 🔍 Project Overview

### Problem Statement
Forecast **next-day gold price returns** (short-term movements) using machine learning and deep learning. Compare effectiveness across three model classes: statistical baselines, traditional ML, and neural networks.

### Data
- **Time Period:** 2003-2024 (21 years, daily)
- **Sources:** Yahoo Finance (gold, SPY), OANDA (EUR/USD), etc.
- **Final Dataset:** 5,469 rows × 43 columns
- **Target:** Next-day gold return

### Features (42 total)
- **Endogenous:** Lagged returns (5), volatility (3), moving averages (3), momentum (3)
- **Technical:** RSI(14), MACD, MACD signal
- **Exogenous:** Lagged EUR/USD, VIX, SPY, Treasury 10Y, DXY, WTI oil (8 features × 3 lags)

### Models Evaluated

**Baselines (Simple benchmarks)**
- Naive: Yesterday's return
- SMA(20): Static 20-day moving average
- ARIMA(1,0,1): Statistical model

**Machine Learning (Traditional)**
- Random Forest: 300 trees, max_depth=6
- XGBoost: 500 trees, learning_rate=0.05

**Deep Learning (Neural networks)**
- MLP: 64→32→1 dense layers
- LSTM: 32 units, 5-timestep sequences
- GRU: 32 units, 5-timestep sequences

### Validation Strategy
- **Train:** 2003-2015 (3,123 samples)
- **Test:** 2016-2024 (9 annual folds)
- **Method:** Walk-forward validation (no lookahead bias)

---

## 🎯 Key Findings

1. **Random Forest is optimal** for magnitude prediction (RMSE=0.00890)
2. **SMA surprisingly competitive** - Nearly matches RF, much simpler
3. **Direction prediction is hard** - All models achieve ~50% (random = 50%)
4. **Deep learning underperforms** - Likely needs more training (epochs=20 too low)
5. **2020 volatility is challenging** - COVID period causes significant errors

### Recommendation
**Deploy Random Forest** for production forecasting. Consider ensemble approach (RF + SMA) for robustness.

---

## 🛠️ Technical Stack

| Component | Tools |
|-----------|-------|
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, xgboost |
| **Deep Learning** | TensorFlow, Keras |
| **Time Series** | statsmodels |
| **Visualization** | matplotlib, seaborn |
| **Automation** | papermill, jupyter |

**Python Version:** 3.8+

---

## 📖 How to Use This Repository

### For Exploration
```bash
# Read docs in order
1. docs/METHODOLOGY.md    (understand approach)
2. docs/RESULTS.md        (see findings)
3. docs/CONCLUSIONS.md    (get recommendations)
```

### For Reproduction
```bash
# Run everything with one command
python scripts/run_all_notebooks.py

# Or run individual notebooks
jupyter notebook notebooks/01_data_collection.ipynb
```

### For Learning
```bash
# Read notebooks in order (01-08)
# Each notebook is well-commented and builds on previous

notebooks/01_data_collection.ipynb        # Data gathering
notebooks/02_data_cleaning.ipynb          # Data preparation
notebooks/03_exploratory_analysis.ipynb   # EDA
notebooks/04_feature_engineering.ipynb    # Feature creation
notebooks/05_baseline_models.ipynb        # Baseline comparison
notebooks/06_ml_models.ipynb              # ML models
notebooks/07_deep_learning.ipynb          # DL models
notebooks/08_walk_forward_validation.ipynb # Final evaluation
```

---

## 📊 Reproducing Results

### Step 1: Environment Setup
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Pipeline
```bash
# Run all notebooks automatically
python scripts/run_all_notebooks.py
```

### Step 3: View Results
Results are saved automatically:
```
results/metrics/walk_forward_results.csv   ← Performance metrics
results/figures/                           ← Visualizations
docs/RESULTS.md                            ← Summary report
```

### Step 4: Examine Notebooks
View individual notebooks for detailed analysis:
```bash
jupyter notebook notebooks/08_walk_forward_validation.ipynb
```

---

## 📈 Results Summary

### Performance by Year
See detailed year-by-year breakdown in [docs/RESULTS.md](docs/RESULTS.md)

### Model Rankings
**By RMSE (Magnitude Prediction):**
1. Random Forest: 0.00890
2. SMA(20): 0.00891
3. XGBoost: 0.00929

**By Directional Accuracy:**
1. GRU: 51.11%
2. Random Forest: 50.81%
3. SMA(20): 50.46%

---

## 💡 Key Insights

### Why are Baselines Competitive?
Gold prices exhibit **mean-reversion patterns**. A simple 20-day SMA performs nearly as well as complex ML models, suggesting the market may be close to **efficient** for daily forecasts.

### Why is Direction Hard to Predict?
Directional accuracy ~50% (random guess) suggests gold returns are close to **random walk** for direction. Magnitude prediction is more feasible than timing.

### Why Does MLP Underperform?
- Training time: 20 epochs (likely insufficient)
- Architecture: Simple feedforward may not capture temporal patterns
- **Solution:** Increase epochs to 50+, add regularization

### 2020: The COVID Challenge
Extreme volatility during COVID (March-April 2020) caused:
- MLP RMSE: 0.1245 (vs avg 0.0270)
- Other models: More robust (RF: 0.0138)

---

## 🔬 Methodology Highlights

### No Data Leakage
- All lagged features created during feature engineering
- Test set never seen during training
- Walk-forward validation ensures realistic performance

### Fair Comparison
- All models tested on identical train/test splits
- Same feature set (42 features)
- Consistent evaluation metrics

### Proper Validation
- **Static split (Notebooks 05-07):** Train 2003-2015, test 2016
- **Walk-forward (Notebook 08):** Expanding windows 2016-2024
- **Metrics:** RMSE, MAE, Directional Accuracy

---

## 📝 Future Work

### Model Improvements
- Increase neural network training (50+ epochs)
- Add attention mechanisms (Transformers)
- Ensemble methods (voting, stacking)
- Hyperparameter tuning

### Data Enhancements
- Include sentiment/news indicators
- Add intraday data
- Incorporate options market data
- Consider transaction costs

### Production Deployment
- Create retraining pipeline
- Implement real-time predictions
- Monitor model drift
- Develop risk management rules

---

## 📄 Citation

If you use this project in your work:

```
@misc{gold_forecasting_capstone,
  author = {[Your Name]},
  title = {Gold Price Forecasting: ML/DL Model Comparison},
  year = {2025},
  url = {https://github.com/yourusername/gold-capstone}
}
```

---

## 🙏 Acknowledgments

- MSCF Program at [University Name]
- Faculty advisor: [Professor Name]
- Data sources: Yahoo Finance, OANDA, etc.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** December 2025  
**Status:** Complete ✅

For questions or feedback, please open an issue or contact [your.email@example.com]
