# How to Structure a Capstone Project from Scratch

## 1. HIGH-LEVEL PROJECT STRUCTURE

```
my-capstone-project/
├── README.md                          # Project entry point
├── setup.py                           # Package metadata
├── requirements.txt                   # Dependencies
├── .gitignore                         # Version control
│
├── docs/                              # Documentation (final deliverables)
│   ├── README.md                      # How to read the docs
│   ├── 01_PROJECT_PROPOSAL.md         # Initial problem statement
│   ├── 02_METHODOLOGY.md              # Data, features, models
│   ├── 03_RESULTS.md                  # Key findings & metrics
│   ├── 04_CONCLUSIONS.md              # Insights & recommendations
│   ├── REPORT.pdf                     # Final capstone report
│   └── figures/                       # All visualizations (.png/.pdf)
│       ├── 01_data_overview.png
│       ├── 02_feature_importance.png
│       ├── 03_model_comparison.png
│       └── ...
│
├── notebooks/                         # Exploratory work (reproducible)
│   ├── README.md                      # Notebook guide
│   ├── 00_project_setup.ipynb         # NEW: Environment setup
│   ├── 01_data_collection.ipynb       # Gather raw data
│   ├── 02_data_cleaning.ipynb         # Handle missing values, outliers
│   ├── 03_eda.ipynb                   # Exploratory analysis
│   ├── 04_feature_engineering.ipynb   # Create features
│   ├── 05_baseline_models.ipynb       # Simple benchmarks
│   ├── 06_ml_models.ipynb             # Machine learning
│   ├── 07_deep_learning.ipynb         # Neural networks
│   ├── 08_validation.ipynb            # Walk-forward, backtesting
│   └── 09_final_summary.ipynb         # Results synthesis
│
├── src/                               # Production-ready code (modules)
│   ├── __init__.py
│   ├── config.py                      # Centralized settings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py                 # Download/load data
│   │   ├── cleaners.py                # Handle NaN, outliers
│   │   └── processors.py              # Data preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py             # Lag, MA, RSI, MACD
│   │   ├── selection.py               # Feature selection
│   │   └── scalers.py                 # Normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py               # Naive, SMA, ARIMA
│   │   ├── ml.py                      # RF, XGB
│   │   └── dl.py                      # MLP, LSTM, GRU
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # RMSE, MAE, DA
│   │   └── validators.py              # Walk-forward logic
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py            # Run full pipeline
│   └── utils/
│       ├── __init__.py
│       ├── io.py                      # Save/load models
│       └── plotting.py                # Visualization helpers
│
├── config/                            # Configuration files
│   ├── default.yaml                   # Global settings
│   ├── models.yaml                    # Model hyperparameters
│   └── paths.yaml                     # Data paths
│
├── scripts/                           # Executable scripts
│   ├── train.py                       # Full training pipeline
│   ├── predict.py                     # Make predictions
│   ├── evaluate.py                    # Run evaluation
│   └── generate_report.py             # Create final report
│
├── data/                              # Data storage (gitignored)
│   ├── raw/                           # Original sources
│   │   ├── gold.csv
│   │   ├── eurusd.csv
│   │   └── ...
│   └── processed/                     # Clean, feature-engineered
│       └── modeling_dataset.csv
│
├── results/                           # Model outputs (gitignored)
│   ├── metrics/
│   │   ├── baseline_results.csv
│   │   ├── ml_results.csv
│   │   └── walk_forward_results.csv
│   ├── models/
│   │   ├── rf_model.pkl
│   │   ├── xgb_model.pkl
│   │   └── lstm_model.h5
│   └── figures/                       # Generated plots
│       ├── rmse_by_year.png
│       ├── directional_accuracy.png
│       └── ...
│
└── tests/                             # Unit tests
    ├── __init__.py
    ├── test_data.py
    ├── test_features.py
    └── test_models.py
```

---

## 2. DETAILED FILE DESCRIPTIONS

### 📄 **ROOT LEVEL**

#### `README.md` (Your project's landing page)
```markdown
# Gold Price Forecasting Capstone

## Overview
[1-2 sentence description of what you're doing]

## Quick Start
```bash
pip install -r requirements.txt
python scripts/train.py
```

## Key Results
- Best model: Random Forest
- RMSE: 0.00890 | MAE: 0.00640 | Dir.Acc: 50.81%
- [See docs/RESULTS.md for full analysis]

## Project Structure
[Brief explanation of folders]

## How to Reproduce
1. Download data: `python scripts/collect_data.py`
2. Train models: `python scripts/train.py`
3. View results: Open `notebooks/09_final_summary.ipynb`
```

#### `setup.py` (Package metadata)
```python
from setuptools import setup, find_packages

setup(
    name='gold-capstone',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Gold price forecasting with ML/DL models',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'tensorflow>=2.8.0',
    ],
)
```

#### `requirements.txt`
```
pandas==1.5.3
numpy==1.24.0
scikit-learn==1.2.0
xgboost==1.7.0
tensorflow==2.12.0
statsmodels==0.13.5
matplotlib==3.7.0
seaborn==0.12.2
pyyaml==6.0
```

---

### 📁 **docs/** - Your Capstone Deliverables

#### `docs/README.md`
```markdown
# Documentation Structure

1. **01_PROJECT_PROPOSAL.md** - Problem statement & objectives
2. **02_METHODOLOGY.md** - Data sources, feature engineering, models used
3. **03_RESULTS.md** - Tables, metrics, rankings
4. **04_CONCLUSIONS.md** - Key findings, recommendations
5. **REPORT.pdf** - Final capstone report (if required)

All figures should be in `figures/` subfolder.
```

#### `docs/01_PROJECT_PROPOSAL.md`
```markdown
# Project Proposal: Gold Price Forecasting

## Problem Statement
Forecast next-day gold price returns using ML/DL models.

## Objectives
1. Compare baseline statistical models vs ML vs DL
2. Identify best-performing model
3. Validate on unseen data (walk-forward validation)

## Data Sources
- Gold prices: [source]
- Macroeconomic indicators: DXY, VIX, SPY, etc.
- Time period: 2003-2024

## Success Metrics
- RMSE, MAE, Directional Accuracy
- Walk-forward validation on 2016-2024
```

#### `docs/02_METHODOLOGY.md`
```markdown
# Methodology

## Data Collection
- 7 data sources merged on date
- Forward-fill for missing values
- Final dataset: 5,469 rows × 43 columns

## Feature Engineering
- Lagged returns (5 lags)
- Technical indicators: RSI, MACD, volatility
- Macroeconomic lags (3 days)
- Target: Next-day gold return

## Models Evaluated
1. **Baselines**: Naive, SMA(20), ARIMA(1,0,1)
2. **ML**: Random Forest, XGBoost
3. **DL**: MLP, LSTM, GRU

## Validation Strategy
- Train: 2003-[year-1]
- Test: [year]
- Applied to 2016-2024 (9 folds)
```

#### `docs/03_RESULTS.md`
```markdown
# Results & Findings

## Model Performance (Average across 2016-2024)

| Model    | RMSE    | MAE     | Dir.Acc |
|----------|---------|---------|---------|
| RF       | 0.00890 | 0.00640 | 50.81%  |
| SMA20    | 0.00891 | 0.00643 | 50.46%  |
| XGB      | 0.00929 | 0.00687 | 47.48%  |
| LSTM     | 0.00937 | 0.00691 | 49.66%  |
| GRU      | 0.00960 | 0.00709 | 51.11%  |
| Naive    | 0.01249 | 0.00945 | 44.97%  |
| MLP      | 0.02699 | 0.02302 | 49.70%  |

[Insert plots here]
```

#### `docs/04_CONCLUSIONS.md`
```markdown
# Conclusions & Recommendations

## Key Findings
1. **Random Forest performs best** - Beats baselines & DL models
2. **SMA surprisingly competitive** - Simple mean-reversion works
3. **Direction prediction hard** - All models ~50% accuracy
4. **MLP struggles** - Insufficient training (20 epochs too low)

## Recommendations
1. **Use Random Forest** for production
2. **Combine predictions** - Ensemble RF + SMA for robustness
3. **Future work**: Increase DL epochs, add attention mechanisms

## Limitations
- Training on 2003-2024 data only
- Daily data may hide intraday patterns
- COVID volatility (2020) challenging for MLP
```

---

### 📔 **notebooks/** - Exploration & Analysis

#### `00_project_setup.ipynb`
```python
# Setup notebook - Run this first
# 1. Import all libraries
# 2. Set up paths
# 3. Load constants from config/
# 4. Verify data availability
```

#### `09_final_summary.ipynb`
```python
# Final synthesis notebook
# 1. Load all results from previous notebooks
# 2. Create comprehensive comparison table
# 3. Generate all final visualizations
# 4. State conclusions & recommendations
# 5. Export to HTML/PDF for presentation
```

---

### 🔧 **src/** - Reusable Production Code

#### `src/config.py`
```python
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Load from YAML
with open(PROJECT_ROOT / 'config' / 'default.yaml') as f:
    CONFIG = yaml.safe_load(f)

TRAIN_START = '2003-01-01'
TEST_START = '2016-01-01'
TEST_END = '2024-12-31'
```

#### `src/data/loaders.py`
```python
def load_raw_data(source: str) -> pd.DataFrame:
    """Load raw CSV files"""
    pass

def load_modeling_dataset() -> pd.DataFrame:
    """Load cleaned, feature-engineered dataset"""
    path = Path('data/processed/modeling_dataset.csv')
    return pd.read_csv(path, index_col=0, parse_dates=True)

def train_test_split(df: pd.DataFrame, year: int):
    """Split by year for walk-forward validation"""
    train = df[df.index < f'{year}-01-01']
    test = df[(df.index >= f'{year}-01-01') & 
              (df.index <= f'{year}-12-31')]
    return train, test
```

#### `src/models/baselines.py`
```python
class NaiveForecaster:
    def predict(self, X_test):
        return X_test['gold_lag1']

class SMAForecaster:
    def __init__(self, window=20):
        self.window = window
        self.sma_value = None
    
    def fit(self, y_train):
        self.sma_value = y_train.rolling(self.window).mean().iloc[-1]
    
    def predict(self, X_test):
        return pd.Series([self.sma_value] * len(X_test))
```

#### `src/evaluation/metrics.py`
```python
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, Directional Accuracy"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    dir_acc = (np.sign(y_true) == np.sign(y_pred)).mean() * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'DirectionalAccuracy': dir_acc
    }
```

#### `src/pipeline/orchestrator.py`
```python
def run_full_pipeline():
    """Execute entire workflow"""
    # 1. Load data
    df = load_modeling_dataset()
    
    # 2. Run walk-forward validation
    results = []
    for year in range(2016, 2025):
        train, test = train_test_split(df, year)
        
        # Train all models
        models = {
            'naive': NaiveForecaster(),
            'sma20': SMAForecaster(window=20),
            'rf': RandomForestRegressor(),
            'lstm': build_lstm()
        }
        
        # Evaluate
        for name, model in models.items():
            metrics = evaluate_model(model, train, test)
            results.append({'Year': year, 'Model': name, **metrics})
    
    # 3. Save & visualize
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/metrics/all_results.csv')
    plot_results(results_df)
```

---

### 📋 **config/** - Centralized Settings

#### `config/default.yaml`
```yaml
# Global project settings
project_name: "Gold Price Forecasting"
version: "1.0.0"

# Data settings
data:
  train_start: "2003-01-01"
  test_start: "2016-01-01"
  test_end: "2024-12-31"
  
# Model hyperparameters
models:
  baseline:
    sma_window: 20
  
  random_forest:
    n_estimators: 300
    max_depth: 6
    random_state: 42
  
  xgboost:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 5
  
  lstm:
    units: 32
    epochs: 50
    batch_size: 32
    timesteps: 5
```

---

### 🚀 **scripts/** - Executable Programs

#### `scripts/train.py`
```python
#!/usr/bin/env python
"""Full training pipeline"""
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import run_full_pipeline

if __name__ == '__main__':
    print("Starting training pipeline...")
    run_full_pipeline()
    print("✓ Pipeline complete. Results saved to results/")
```

#### `scripts/generate_report.py`
```python
#!/usr/bin/env python
"""Generate final capstone report"""
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    results = pd.read_csv('results/metrics/walk_forward_results.csv')
    
    # Summary statistics
    print(results.describe())
    
    # Best models
    print(f"\nBest RMSE: {results.iloc[:, 1:].mean().idxmin()}")
    
    # Export to markdown
    with open('docs/03_RESULTS.md', 'w') as f:
        f.write("# Results\n\n")
        f.write(results.to_markdown())
```

---

## 3. WORKFLOW & EXECUTION ORDER

### **Phase 1: Planning (Before Coding)**
```
1. Create repository structure
2. Write docs/01_PROJECT_PROPOSAL.md
3. Set up config files
4. Create requirements.txt
```

### **Phase 2: Data & Exploration**
```
1. Run notebooks/01_data_collection.ipynb
2. Run notebooks/02_data_cleaning.ipynb
3. Run notebooks/03_eda.ipynb
4. Save processed data
```

### **Phase 3: Modeling**
```
1. Run notebooks/04_feature_engineering.ipynb
2. Run notebooks/05_baseline_models.ipynb
3. Run notebooks/06_ml_models.ipynb
4. Run notebooks/07_deep_learning.ipynb
5. Run notebooks/08_validation.ipynb
```

### **Phase 4: Results & Reporting**
```
1. Run notebooks/09_final_summary.ipynb
2. Write docs/02_METHODOLOGY.md
3. Write docs/03_RESULTS.md
4. Write docs/04_CONCLUSIONS.md
5. Create docs/REPORT.pdf
6. Run scripts/generate_report.py (auto-generate tables)
```

### **Phase 5: Production Code (Refactoring)**
```
1. Extract functions from notebooks → src/
2. Create config files
3. Test with scripts/train.py
4. Write src/pipeline/orchestrator.py
```

---

## 4. KEY PRINCIPLES

### ✅ DO THIS

| ✓ | Principle |
|---|-----------|
| ✓ | **Separate exploration (notebooks) from production (src/)** |
| ✓ | **Centralize configuration in YAML files** |
| ✓ | **Write reusable functions in src/** |
| ✓ | **Document everything: what, why, how** |
| ✓ | **Version control notebooks (but not data/results)** |
| ✓ | **Use meaningful file & variable names** |
| ✓ | **Create a 09_final_summary notebook** |
| ✓ | **Write comprehensive README** |

### ❌ DON'T DO THIS

| ✗ | Anti-Pattern |
|---|---|
| ✗ | **Keep all code in notebooks only** |
| ✗ | **Hardcode paths** |
| ✗ | **Use generic names like `notebook1.ipynb`** |
| ✗ | **Leave docs empty** |
| ✗ | **Forget to explain your choices** |
| ✗ | **Skip validation strategy** |
| ✗ | **Mix raw and processed data** |

---

## 5. YOUR CURRENT PROJECT - HOW TO REFACTOR

Since you already have notebooks working, here's how to reorganize:

```bash
# 1. Create src/ structure
mkdir -p src/{data,features,models,evaluation,pipeline,utils}
touch src/__init__.py src/{data,features,models,evaluation,pipeline,utils}/__init__.py

# 2. Create config/
mkdir -p config
# Add default.yaml and models.yaml

# 3. Create scripts/
mkdir -p scripts
# Add train.py, evaluate.py, generate_report.py

# 4. Enhance docs/
# Write METHODOLOGY.md, RESULTS.md, CONCLUSIONS.md

# 5. Extract notebook code → src/
# - data loading → src/data/loaders.py
# - feature engineering → src/features/engineering.py
# - model training → src/models/*.py
# - metrics → src/evaluation/metrics.py

# 6. Create notebook 09
# - Load all results
# - Create final comparison table
# - Write conclusions

# 7. Update README
# - Add quick start guide
# - Link to docs/
# - Highlight key results
```

---

## 6. MINIMAL EXAMPLE - Get Started Today

If you want to start refactoring **right now**, minimum steps:

```bash
# Step 1: Create structure
mkdir -p src config scripts
touch src/__init__.py src/config.py

# Step 2: Create config/default.yaml
cat > config/default.yaml << 'EOF'
train_start: "2003-01-01"
test_start: "2016-01-01"
models:
  rf:
    n_estimators: 300
    max_depth: 6
EOF

# Step 3: Create src/config.py
cat > src/config.py << 'EOF'
from pathlib import Path
import yaml
PROJECT_ROOT = Path(__file__).parent.parent
with open(PROJECT_ROOT / 'config/default.yaml') as f:
    CONFIG = yaml.safe_load(f)
EOF

# Step 4: Create scripts/train.py (calls orchestrator)
cat > scripts/train.py << 'EOF'
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.orchestrator import run_full_pipeline
if __name__ == '__main__':
    run_full_pipeline()
EOF

# Step 5: Create README.md with quick start
# Step 6: Create docs/ with METHODOLOGY.md, RESULTS.md, CONCLUSIONS.md
```

---

## 7. FINAL CHECKLIST

Before submission:

- [ ] Comprehensive README.md
- [ ] docs/METHODOLOGY.md complete
- [ ] docs/RESULTS.md with tables & figures
- [ ] docs/CONCLUSIONS.md with findings
- [ ] Notebook 09 (final summary)
- [ ] config/ files created
- [ ] src/ modules extracted
- [ ] scripts/train.py executable
- [ ] requirements.txt complete
- [ ] .gitignore excludes data/ and results/
- [ ] All notebooks run without errors
- [ ] Figures saved to docs/figures/
- [ ] Code has docstrings

---

**TL;DR**: Structure around **deliverables (docs/)** not just notebooks. Separate exploration from production code. Centralize configuration. Document everything.
