# Notebooks Guide

This directory contains the complete analysis pipeline for gold price forecasting. **Run notebooks in order (01 → 08)**.

---

## Execution Guide

### Quick Start
```bash
# Option 1: Run all notebooks automatically
python ../scripts/run_all_notebooks.py

# Option 2: Run individually in Jupyter
jupyter notebook
# Open each notebook in order, cell-by-cell
```

**Expected total runtime:** ~45 minutes (GPU: ~20 minutes)

---

## Notebook Descriptions

### 1️⃣ **01_data_collection.ipynb** (5 min)
**Purpose:** Load raw data from 7 sources

**Input:**
- `../data/raw/dxy.csv`
- `../data/raw/eurusd.csv`
- `../data/raw/gold.csv`
- `../data/raw/oil.csv`
- `../data/raw/spy.csv`
- `../data/raw/treasury_10y.csv`
- `../data/raw/vix.csv`

**Output:**
- Merged time-series (5,732 rows × 7 columns)
- All dates aligned
- Print data summary (shape, dtypes, first/last rows)

**Key Operations:**
```python
- pd.read_csv() for each source
- Merge all on date index
- Handle date alignment across sources
```

**Expected Output:**
```
Data Shape: (5732, 7)
Date Range: 2003-01-03 to 2024-12-27
All columns numeric
No NaN in raw merge
```

---

### 2️⃣ **02_data_cleaning.ipynb** (8 min)
**Purpose:** Clean data and prepare for feature engineering

**Input:**
- Merged data from Notebook 01 (in-memory)

**Process:**
1. Forward-fill missing values (handles gaps in trading)
2. Compute returns (log returns for most, simple returns preserved)
3. Identify and document outliers
4. Remove rows with remaining NaN

**Output:**
- `../data/processed/cleaned_data.csv` (5,473 rows × 7 columns)
- Cleaned data object for next notebook
- Summary statistics and data quality report

**Key Statistics:**
```
Starting rows: 5,732
After cleaning: 5,469 (99.5% retention)
Missing values: 0
Outliers identified: None after forward-fill
```

**Expected Output:**
```
Data Quality Report:
- Duplicates: 0
- Missing values: 0
- Data types: All float64
- Date range: 2003-01-03 to 2024-12-27
```

---

### 3️⃣ **03_exploratory_analysis.ipynb** (12 min)
**Purpose:** Understand data distributions and relationships

**Input:**
- `../data/processed/cleaned_data.csv`

**Analysis:**
1. Univariate statistics (mean, std, skewness, kurtosis)
2. Distribution plots (histograms, KDE)
3. Correlation heatmap (7×7 assets)
4. Time-series plots
5. Rolling correlations
6. Autocorrelation (ACF) and partial autocorrelation (PACF)

**Output:**
- Multiple visualizations (saved to `../docs/figures/` if specified)
- Correlation matrix printed
- Autocorrelation summary
- Key findings documented

**Key Findings Expected:**
```
Gold return mean: ~0.03% daily
Gold return std: ~1.2% daily
Correlation gold-SPY: ~0.15 (weak)
Correlation gold-VIX: ~0.25 (weak)
ACF lag-1: ~0.08 (weak autocorrelation)
```

**Expected Output:**
```
✓ 7 distribution plots
✓ 1 correlation heatmap
✓ 2 ACF/PACF plots
✓ 2 rolling correlation plots
✓ Summary statistics table
```

---

### 4️⃣ **04_feature_engineering.ipynb** (10 min)
**Purpose:** Create 42 features for modeling

**Input:**
- `../data/processed/cleaned_data.csv`

**Features Created (42 total):**

**Endogenous (Gold-based) - 17 features:**
- Lags: gold_lag1, gold_lag2, gold_lag3, gold_lag4, gold_lag5
- Volatility: vol_10, vol_20, vol_30 (rolling std)
- Moving Averages: ma_10, ma_20, ma_30
- Momentum: momentum_10, momentum_20, momentum_30
- Technical: rsi_14, macd, macd_signal

**Exogenous (Macro) - 24 features:**
- eurusd_lag1, eurusd_lag2, eurusd_lag3
- treasury_10y_lag1, treasury_10y_lag2, treasury_10y_lag3
- spy_lag1, spy_lag2, spy_lag3
- vix_lag1, vix_lag2, vix_lag3
- dxy_lag1, dxy_lag2, dxy_lag3
- oil_lag1, oil_lag2, oil_lag3

**Target - 1 feature:**
- next_day_gold_return (shift(-1) for no lookahead bias)

**Output:**
- `../data/processed/modeling_dataset.csv` (5,469 rows × 43 columns)
- Features summary table
- Feature engineering documentation

**Data Quality Check:**
```
Starting rows: 5,473
After feature creation: 5,469 (dropped NaN from shifts)
Total features: 42 + 1 target = 43 columns
Missing values: 0
```

**Expected Output:**
```
✓ modeling_dataset.csv created
✓ 43 columns (42 features + 1 target)
✓ 5,469 rows
✓ 0 missing values
✓ Feature summary table printed
```

---

### 5️⃣ **05_baseline_models.ipynb** (8 min)
**Purpose:** Train and evaluate 3 baseline models

**Input:**
- `../data/processed/modeling_dataset.csv`

**Models Trained:**

1. **Naive Baseline**
   ```python
   prediction = X_test['gold_lag1']
   ```
   - Method: Yesterday's return repeated

2. **SMA(20) Baseline**
   ```python
   sma = y_train.rolling(20).mean().iloc[-1]
   prediction = [sma] * len(test)
   ```
   - Method: Static mean from training period

3. **ARIMA(1,0,1)**
   ```python
   ARIMA(y_train, order=(1,0,1)).fit().forecast(len(test))
   ```
   - Method: Statistical autoregressive model

**Train-Test Split:**
```
Train: 2003-01-03 to 2015-12-31 (3,123 samples)
Test:  2016-01-01 to 2016-12-30 (261 samples)
```

**Output:**
- RMSE, MAE, Directional Accuracy for each baseline
- Predictions vs. actuals plot
- Baseline comparison table

**Expected Results:**
```
Naive RMSE:    0.0139
SMA RMSE:      0.0101 ← Best baseline
ARIMA RMSE:    0.0100
```

**Expected Output:**
```
✓ Baseline comparison table
✓ Predictions vs actuals plot
✓ Metrics for each model (RMSE, MAE, DA)
✓ Conclusion: SMA surprisingly competitive
```

---

### 6️⃣ **06_ml_models.ipynb** (15 min)
**Purpose:** Train and evaluate 2 machine learning models

**Input:**
- `../data/processed/modeling_dataset.csv`

**Models Trained:**

1. **Random Forest**
   ```python
   RandomForestRegressor(
       n_estimators=300,
       max_depth=6,
       random_state=42
   )
   ```
   - Input: StandardScaler normalized features
   - Training: 3,123 samples

2. **XGBoost**
   ```python
   XGBRegressor(
       n_estimators=500,
       learning_rate=0.05,
       max_depth=5,
       subsample=0.8,
       colsample_bytree=0.8
   )
   ```
   - Input: Unscaled features (tree-based)
   - Training: 3,123 samples

**Train-Test Split:**
```
Same as baseline: 2003-2015 train, 2016 test
```

**Output:**
- RMSE, MAE, Directional Accuracy
- Feature importance (top 20 features)
- Feature importance plot
- Predictions vs. actuals
- ML comparison with baselines

**Expected Results:**
```
RF RMSE:  0.0072 (Best!)
XGB RMSE: 0.0076
```

**Expected Output:**
```
✓ RF feature importance bar plot
✓ Top 10 features: gold_lag1, vol_20, gold_lag2, ...
✓ Predictions plot (RF vs XGB)
✓ Model comparison table
✓ Conclusion: RF best non-baseline model
```

---

### 7️⃣ **07_deep_learning.ipynb** (25 min)
**Purpose:** Train and evaluate 2 deep learning models

**Input:**
- `../data/processed/modeling_dataset.csv`

**Models Trained:**

1. **MLP (Multi-Layer Perceptron)**
   ```python
   Sequential([
       Dense(64, activation='relu', input_shape=(42,)),
       Dense(32, activation='relu'),
       Dense(1)
   ])
   optimizer='adam', loss='mse'
   epochs=20, batch_size=32
   ```
   - Input: 42 features (StandardScaler normalized)
   - Training: 3,123 samples

2. **LSTM (Long Short-Term Memory)**
   ```python
   Sequential([
       LSTM(32, return_sequences=False, input_shape=(5, 42)),
       Dense(1)
   ])
   optimizer='adam', loss='mse'
   epochs=40, batch_size=32
   ```
   - Input: (5 timesteps, 42 features) reshaped sequences
   - Training: 3,119 samples (5 fewer due to sequence creation)

3. **GRU (Gated Recurrent Unit)**
   ```python
   Sequential([
       GRU(32, return_sequences=False, input_shape=(5, 42)),
       Dense(1)
   ])
   optimizer='adam', loss='mse'
   epochs=40, batch_size=32
   ```
   - Input: (5 timesteps, 42 features) reshaped sequences
   - Training: 3,119 samples

**Train-Test Split:**
```
Train: 2003-2015 (3,123 or 3,119 samples)
Test:  2016 (261 or 257 samples)
```

**Output:**
- RMSE, MAE, Directional Accuracy for each DL model
- Training history plots (loss over epochs)
- Predictions vs. actuals
- DL model comparison
- Comparison with ML and baselines

**Expected Results:**
```
MLP RMSE:  0.0110 (Worst)
LSTM RMSE: 0.0096
GRU RMSE:  0.0098 (but best DA: 51.88%)
```

**Expected Output:**
```
✓ 3 training history plots (loss curves)
✓ Predictions plot (MLP vs LSTM vs GRU)
✓ Model comparison table (all models)
✓ Conclusion: DL worse than RF, but GRU best direction
```

---

### 8️⃣ **08_walk_forward_validation.ipynb** ⭐ **MAIN NOTEBOOK** (20 min)
**Purpose:** Validate all 7 models across 9 years using walk-forward methodology

**Input:**
- `../data/processed/modeling_dataset.csv`

**Methodology:**
```
For each year Y in [2016, 2017, ..., 2024]:
    Train all 7 models on data from 2003 to Y-1
    Test on all data from year Y
    Compute RMSE, MAE, Directional Accuracy
    Save results
```

**Models Tested:**
1. Naive baseline
2. SMA(20) baseline
3. ARIMA(1,0,1)
4. Random Forest
5. XGBoost
6. MLP (DL)
7. LSTM (DL)
8. GRU (DL)

**Output:**
- `../results/metrics/walk_forward_results.csv` (9 rows × 21 metrics)
  - Row for each year 2016-2024
  - Columns: RMSE, MAE, DA for each of 7 models
  
- `../results/figures/walk_forward_rmse.png`
  - Line plot: RMSE over years (all models)
  
- `../results/figures/walk_forward_da.png`
  - Line plot: Directional Accuracy over years (all models)

- Summary statistics
  - Average RMSE, MAE, DA across 2016-2024
  - Best model by year
  - Overall winner announcement

**Walk-Forward Schedule:**
```
Year 2016: Train 2003-2015 (3,123), Test 2016 (261)
Year 2017: Train 2003-2016 (3,382), Test 2017 (252)
Year 2018: Train 2003-2017 (3,636), Test 2018 (251)
Year 2019: Train 2003-2018 (3,889), Test 2019 (261)
Year 2020: Train 2003-2019 (4,151), Test 2020 (253) ← COVID crisis
Year 2021: Train 2003-2020 (4,404), Test 2021 (261)
Year 2022: Train 2003-2021 (4,667), Test 2022 (261)
Year 2023: Train 2003-2022 (4,928), Test 2023 (261)
Year 2024: Train 2003-2023 (5,189), Test 2024 (280)
```

**Expected Results (2016-2024 Average):**
```
RMSE Rankings:
1. RF:   0.00890 ⭐ Best
2. SMA:  0.00891
3. XGB:  0.00910
4. ARIMA: 0.00927
5. LSTM: 0.00960
6. GRU:  0.00975
7. MLP:  0.01100

Directional Accuracy Rankings:
1. GRU:  51.11% ⭐ Best
2. ARIMA: 48.95%
3. XGB:  48.92%
4. MLP:  49.63%
5. LSTM: 50.00%
6. RF:   48.84%
7. SMA:  47.68%
```

**Key Findings:**
```
✓ RF wins on magnitude (RMSE)
✓ GRU best on direction (DA 51% ≈ random)
✓ SMA baseline competitive (0.01% worse than RF!)
✓ 2020 COVID: All models degraded (GRU dropped to 45% DA)
✓ DL doesn't beat traditional ML for daily returns
```

**Expected Output:**
```
✓ walk_forward_results.csv (9 years × 7 models)
✓ RMSE line plot showing all models over 2016-2024
✓ Directional Accuracy line plot
✓ Yearly best-model summary table
✓ Average metrics summary table
✓ Key findings printed to console
```

---

## Dependency Chain

```
01_data_collection.ipynb
    ↓ (output: merged data)
02_data_cleaning.ipynb
    ↓ (output: cleaned_data.csv)
03_exploratory_analysis.ipynb
    ↓ (uses cleaned data, no output needed for next steps)
04_feature_engineering.ipynb
    ↓ (output: modeling_dataset.csv) ← CRITICAL
    ├→ 05_baseline_models.ipynb
    ├→ 06_ml_models.ipynb
    ├→ 07_deep_learning.ipynb
    └→ 08_walk_forward_validation.ipynb
        └ (final comprehensive validation)
```

**Key Dependencies:**
- **Must run 01-04 first** (data preparation)
- **Can run 05-07 in parallel** (independent baseline/ML/DL)
- **Must run 08 last** (uses trained models and walks forward)

---

## How to Run

### Option 1: Automatic (All Notebooks)
```bash
cd /Users/erion/Desktop/MSCF2/ADA/Capstone-Project
python scripts/run_all_notebooks.py
```
- Runs all 8 notebooks in order
- Generates outputs automatically
- Total time: ~45 minutes

### Option 2: Manual (Jupyter)
```bash
jupyter notebook

# In Jupyter:
# 1. Open 01_data_collection.ipynb
# 2. Run all cells (Shift+Ctrl+Enter)
# 3. Repeat for 02, 03, 04, 05, 06, 07, 08
```

### Option 3: Visual Studio Code
```bash
code /Users/erion/Desktop/MSCF2/ADA/Capstone-Project

# In VS Code:
# 1. Open notebook 01_data_collection.ipynb
# 2. Select Python kernel
# 3. Run all cells
# 4. Repeat for each notebook
```

---

## Expected Outputs

### By Notebook
| Notebook | CSV Output | PNG Output | Console Output |
|----------|-----------|-----------|----------------|
| 01 | None | None | Data shape summary |
| 02 | cleaned_data.csv | None | Data quality report |
| 03 | None | EDA plots (optional) | Stats table |
| 04 | modeling_dataset.csv | None | Feature summary |
| 05 | None | baseline_comparison.png | Metrics table |
| 06 | None | feature_importance.png | ML metrics |
| 07 | None | training_history.png | DL metrics |
| 08 | walk_forward_results.csv | rmse_plot.png, da_plot.png | Summary table |

### File Locations
```
data/
  processed/
    cleaned_data.csv ← From 02
    modeling_dataset.csv ← From 04

results/
  metrics/
    walk_forward_results.csv ← From 08
  figures/
    [all PNG plots from 05, 06, 07, 08]
```

---

## Troubleshooting

### Issue: ImportError for pandas, numpy, etc.
**Solution:** Install requirements
```bash
pip install -r ../requirements.txt
```

### Issue: Notebook runs but produces no output
**Solution:** Check if cells are executed
- Look for `[*]` (running) vs `[1]` (executed)
- Click "Run All" or press Shift+Ctrl+Enter

### Issue: slow_dataset.csv not found
**Solution:** Run notebook 04 first (creates this file)
```bash
# Make sure to run in order:
01 → 02 → 03 → 04 → (05, 06, 07) → 08
```

### Issue: LSTM/GRU takes too long (>5 min per notebook)
**Solution:** Check if using GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# If empty, using CPU (slower)
# Install GPU support or reduce epochs in code
```

### Issue: Results don't match documentation
**Solution:** Check random seed
```python
# All notebooks should set:
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

---

## Summary

**Run order:** 01 → 02 → 03 → 04 → 05-07 (parallel OK) → 08

**Key outputs:**
- Notebook 04: `modeling_dataset.csv` (features)
- Notebook 08: `walk_forward_results.csv` (final results)

**Expected runtime:** 45 minutes total

**Success criteria:**
- All notebooks run without errors
- walk_forward_results.csv has 9 rows (one per year)
- RMSE/MAE/DA values reasonable (0.0089 RMSE ± 0.0005)

---

For detailed methodology, see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md)
For detailed results, see [docs/RESULTS.md](../docs/RESULTS.md)
For conclusions, see [docs/CONCLUSIONS.md](../docs/CONCLUSIONS.md)
