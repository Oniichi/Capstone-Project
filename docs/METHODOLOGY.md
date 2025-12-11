# Methodology

## Data Collection & Preparation

### Data Sources
- **Gold Prices:** Yahoo Finance (daily close)
- **Currency (EUR/USD):** OANDA
- **Equity Index (SPY):** Yahoo Finance
- **Volatility (VIX):** Yahoo Finance
- **Treasury 10Y:** Yahoo Finance
- **USD Index (DXY):** Yahoo Finance
- **Crude Oil (WTI):** Yahoo Finance

### Time Period
- **Start:** January 3, 2003
- **End:** December 27, 2024
- **Duration:** 21 years of daily data
- **Total trading days:** 5,732 (before cleaning)

### Data Cleaning
1. **Merged all 7 time series** by trading date
2. **Handled missing values** with forward-fill (99.5% retention)
3. **Removed rows with missing values** after feature engineering
4. **Final dataset:** 5,469 rows × 43 columns

### Target Variable
```python
target = gold_return_tomorrow
target = df['gold'].shift(-1)  # Next-day return (no lookahead bias)
```

---

## Feature Engineering

### Feature Set (42 total)

#### 1. Endogenous Features (Gold-Based) - 17 features

**Lagged Returns (5 features)**
```
gold_lag1, gold_lag2, gold_lag3, gold_lag4, gold_lag5
```
Previous 5 days of gold returns (momentum)

**Volatility (3 features)**
```
vol_10 = std(gold_returns, window=10)
vol_20 = std(gold_returns, window=20)
vol_30 = std(gold_returns, window=30)
```
Rolling standard deviation to capture market volatility

**Moving Averages (3 features)**
```
ma_10 = mean(gold, window=10)
ma_20 = mean(gold, window=20)
ma_30 = mean(gold, window=30)
```
Trend indicators

**Momentum (3 features)**
```
momentum_10 = sum(gold_returns, window=10)
momentum_20 = sum(gold_returns, window=20)
momentum_30 = sum(gold_returns, window=30)
```
Accumulated returns over periods

**Technical Indicators (3 features)**
```
rsi_14 = RSI(gold, period=14)          # Relative Strength Index
macd = ema12 - ema26                   # MACD line
macd_signal = ema(macd, span=9)        # Signal line
```

#### 2. Exogenous Features (Macroeconomic) - 24 features

**7 Macro Time Series, Each with 3-day lags:**
- eurusd_lag1, eurusd_lag2, eurusd_lag3
- treasury_10y_lag1, treasury_10y_lag2, treasury_10y_lag3
- spy_lag1, spy_lag2, spy_lag3
- vix_lag1, vix_lag2, vix_lag3
- dxy_lag1, dxy_lag2, dxy_lag3
- oil_lag1, oil_lag2, oil_lag3

**Rationale:** Captures delayed market responses and dependencies

#### 3. Target Variable (1)
```
target = next_day_gold_return
```

### No Data Leakage
- ✅ All features use past information only
- ✅ Lagged features created during preprocessing
- ✅ Test set never seen during feature engineering
- ✅ Walk-forward validation prevents lookahead bias

---

## Train-Test Splitting

### Static Split (Notebooks 05-07)
```
Train: 2003-01-13 to 2015-12-31 (3,123 samples)
Test:  2016-01-01 to 2016-12-30 (261 samples)
```
Used to establish baseline performance on single year.

### Walk-Forward Validation (Notebook 08)
```
Year 2016: Train 2003-2015 (3,123),  Test 2016 (261)
Year 2017: Train 2003-2016 (3,382),  Test 2017 (252)
Year 2018: Train 2003-2017 (3,636),  Test 2018 (251)
Year 2019: Train 2003-2018 (3,889),  Test 2019 (261)
Year 2020: Train 2003-2019 (4,151),  Test 2020 (253)
Year 2021: Train 2003-2020 (4,404),  Test 2021 (261)
Year 2022: Train 2003-2021 (4,667),  Test 2022 (261)
Year 2023: Train 2003-2022 (4,928),  Test 2023 (261)
Year 2024: Train 2003-2023 (5,189),  Test 2024 (280)
```
Expands training window, tests on subsequent year (9 folds).

---

## Models Evaluated

### 1. Baseline Models

#### Naive Baseline
```python
prediction = X_test['gold_lag1']
```
**Logic:** Yesterday's return = today's best guess (simplest possible)

#### SMA(20) Baseline
```python
sma_value = y_train.rolling(20).mean().iloc[-1]
prediction = [sma_value] * len(test)
```
**Logic:** Predict static 20-day moving average from training period

#### ARIMA(1,0,1)
```python
ARIMA(y_train, order=(1,0,1)).fit().forecast(steps=len(test))
```
**Logic:** Autoregressive statistical model with lag-1 and MA-1

### 2. Machine Learning Models

#### Random Forest
```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
```
- **Trees:** 300
- **Max Depth:** 6 (prevents overfitting)
- **Input:** StandardScaler-normalized features

**Why this config?**
- 300 trees balances bias-variance
- max_depth=6 acts as regularization
- Scaling improves consistency

#### XGBoost
```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
```
- **Trees:** 500
- **Learning Rate:** 0.05 (conservative, prevents overfitting)
- **Subsample:** 0.8 (uses 80% of samples per iteration)
- **Input:** Unscaled features (tree-based)

**Why this config?**
- Gradient boosting captures non-linear relationships
- Low learning rate ensures stability
- Subsampling provides regularization

### 3. Deep Learning Models

#### MLP (Multi-Layer Perceptron)
```python
Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output
])
optimizer='adam', loss='mse'
epochs=20, batch_size=32
```

**Architecture:**
- Input: 42 features
- Hidden 1: 64 neurons (ReLU)
- Hidden 2: 32 neurons (ReLU)
- Output: 1 (regression)

#### LSTM (Long Short-Term Memory)
```python
Sequential([
    LSTM(32, return_sequences=False, input_shape=(5, 42)),
    Dense(1)
])
optimizer='adam', loss='mse'
epochs=40, batch_size=32
```

**Architecture:**
- Input: (5 timesteps, 42 features)
- LSTM: 32 units (captures temporal dependencies)
- Output: 1 (regression)

**Why LSTM?**
- Recurrent architecture captures temporal patterns
- Long/short-term memory mechanisms prevent vanishing gradients

#### GRU (Gated Recurrent Unit)
```python
Sequential([
    GRU(32, return_sequences=False, input_shape=(5, 42)),
    Dense(1)
])
optimizer='adam', loss='mse'
epochs=40, batch_size=32
```

**Architecture:**
- Input: (5 timesteps, 42 features)
- GRU: 32 units (simpler than LSTM, fewer parameters)
- Output: 1 (regression)

**Why GRU?**
- Computationally more efficient than LSTM
- Similar performance with fewer parameters

---

## Evaluation Metrics

### 1. Root Mean Squared Error (RMSE)
```
RMSE = sqrt(mean((y_true - y_pred)^2))
```
- **Lower is better**
- Penalizes large errors more (quadratic)
- Range: Unbounded
- Units: Same as target (gold returns)

### 2. Mean Absolute Error (MAE)
```
MAE = mean(|y_true - y_pred|)
```
- **Lower is better**
- Robust to outliers
- Interpretable (avg absolute error)
- Units: Same as target

### 3. Directional Accuracy
```
DA = (sum(sign(y_true) == sign(y_pred)) / len(y_true)) * 100%
```
- **Higher is better**
- Percentage of correct up/down predictions
- Random baseline: 50%
- Range: 0-100%

---

## Validation Strategy

### Why Walk-Forward Validation?
1. **Realistic:** Models train on available data, test on future
2. **No lookahead:** Never uses future data for training
3. **Expanding window:** Simulates production scenarios
4. **Fair comparison:** All models evaluated on identical splits

### Cross-Temporal Validation
- **Baseline (NB 05-07):** Single year (2016) establishes initial performance
- **Walk-forward (NB 08):** 9 folds across market conditions (2016-2024)
- **COVID period (2020):** Includes extreme volatility for robustness test

---

## Data Quality & Assumptions

### Missing Data Handling
- Forward-fill missing values in raw data
- Drop rows with remaining NaN after feature engineering
- Final retention rate: 99.5%

### Stationarity Assumptions
- **Target:** Returns are stationary (not price levels)
- **Features:** Mix of lagged returns (stationary) and raw levels (non-stationary)
- **Implication:** Model captures mean-reversion and volatility patterns

### Market Efficiency Assumptions
- Daily returns may be hard to predict (EMH)
- Magnitude might be more predictable than direction
- Macro variables may have delayed impact (reflected in lags)

---

## Summary Table

| Component | Details |
|-----------|---------|
| **Data Period** | 2003-2024 (21 years) |
| **Final Rows** | 5,469 (99.5% retention) |
| **Features** | 42 (17 endogenous, 24 exogenous, 1 target) |
| **Train-Test Split** | 2003-2015 : 2016-2024 |
| **Validation Method** | Walk-forward (9 folds) |
| **Models** | 7 (3 baselines, 2 ML, 2 DL) |
| **Evaluation Metrics** | RMSE, MAE, Directional Accuracy |
| **No Lookahead Bias** | ✅ Confirmed |

---

**For detailed implementation, see [Notebook 04](../notebooks/04_feature_engineering.ipynb) for feature engineering and [Notebook 08](../notebooks/08_walk_forward_validation.ipynb) for walk-forward validation.**
