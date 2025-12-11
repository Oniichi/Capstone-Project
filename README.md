# Gold Price Forecasting: ML/DL Model Comparison

Comparing baseline statistical models, machine learning, and deep learning approaches for predicting next-day gold price returns using 21 years of data (2003-2024) with walk-forward validation.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Notebooks
```bash
python scripts/run_all_notebooks.py
```

### 3. View Documentation
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — Data, features, models, validation
- **[docs/RESULTS.md](docs/RESULTS.md)** — Performance metrics and analysis
- **[docs/CONCLUSIONS.md](docs/CONCLUSIONS.md)** — Key findings and recommendations

---

## Results

| Model | RMSE | MAE | Direction Accuracy |
|-------|------|-----|-------------------|
| **Random Forest** | **0.00890** | **0.00640** | 50.81% |
| SMA(20) | 0.00891 | 0.00643 | 50.46% |
| XGBoost | 0.00929 | 0.00687 | 47.48% |
| LSTM | 0.00937 | 0.00691 | 49.66% |
| GRU | 0.00960 | 0.00709 | 51.11% |
| Naive | 0.01249 | 0.00945 | 44.97% |
| MLP | 0.02699 | 0.02302 | 49.70% |

**Key Finding:** Random Forest achieves best magnitude prediction (RMSE 0.00890), but all models struggle with direction prediction (~50%, equivalent to random guessing).

---

## Project Structure

```
├── notebooks/              # 8 analysis notebooks (01-08)
│   └── README.md          # Notebook guide
├── docs/                  # Documentation
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── CONCLUSIONS.md
├── scripts/               # Automation
│   └── run_all_notebooks.py
├── data/                  # Data (gitignored)
│   ├── raw/
│   └── processed/
├── results/               # Outputs (gitignored)
│   ├── metrics/
│   └── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset

- **Time Period:** 2003-2024 (21 years, daily)
- **Features:** 42 engineered features (lags, technical, macro)
- **Rows:** 5,469
- **Target:** Next-day gold return

---

## Models

**Baselines:** Naive, SMA(20), ARIMA  
**ML:** Random Forest, XGBoost  
**DL:** MLP, LSTM, GRU

Evaluated using walk-forward validation (2016-2024, 9 annual folds).

---

## Key Insights

1. **Magnitude is predictable** — RF achieves 0.00890 RMSE (vs 0.01388 naive)
2. **Direction is not** — All models ~50% (random baseline)
3. **Simpler is better** — SMA baseline nearly matches RF
4. **Deep learning underperforms** — DL models worse than RF
5. **2020 challenges DL** — COVID volatility breaks neural networks

---

## Recommendation

Deploy **Random Forest** for production. Consider **ensemble (RF + SMA)** for robustness.

---

## License

MIT License - see [LICENSE](LICENSE)
