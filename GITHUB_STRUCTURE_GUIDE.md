# Best Capstone Project Structure for GitHub

## The Goal
Structure your capstone so that:
1. **Anyone** can understand the project in 2 minutes
2. **Anyone** can reproduce your results with one command
3. **Professors/reviewers** see professional organization
4. **GitHub** showcases your work beautifully

---

## The Optimal Structure

```
gold-capstone-project/
│
├── README.md                          # ⭐ MOST IMPORTANT - Landing page
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT or Apache 2.0
├── .gitignore                         # Exclude data/, results/
│
├── docs/                              # 📖 Documentation (what you found)
│   ├── README.md                      # "Start here for docs"
│   ├── PROJECT_PROPOSAL.md            # Problem statement
│   ├── METHODOLOGY.md                 # Data, features, models
│   ├── RESULTS.md                     # Key findings, tables
│   ├── CONCLUSIONS.md                 # Insights & recommendations
│   └── figures/                       # All visualizations
│       ├── 01_data_overview.png
│       ├── 02_rmse_comparison.png
│       ├── 03_feature_importance.png
│       └── 04_model_rankings.png
│
├── notebooks/                         # 📓 Analysis (how you did it)
│   ├── README.md                      # "How to read notebooks"
│   ├── 00_project_setup.ipynb         # Environment setup (RUN FIRST)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_ml_models.ipynb
│   ├── 07_deep_learning.ipynb
│   ├── 08_walk_forward_validation.ipynb
│   └── 09_final_summary.ipynb
│
├── scripts/                           # 🚀 Automation
│   ├── run_all_notebooks.py           # One command to run everything
│   └── run_specific_notebook.py       # Run individual notebooks
│
├── data/                              # 📊 Data (gitignored, but document structure)
│   ├── .gitkeep                       # Keep folder in git even when empty
│   ├── raw/                           # Original data sources
│   │   └── README.md                  # "How to get data"
│   └── processed/                     # Cleaned, engineered data
│       └── .gitkeep
│
├── results/                           # 📈 Model outputs (gitignored)
│   ├── .gitkeep
│   ├── metrics/                       # CSV files with metrics
│   ├── models/                        # Saved model files
│   └── figures/                       # Generated plots
│
├── config/                            # ⚙️ Settings (optional)
│   └── params.yaml                    # Hyperparameters
│
└── .github/                           # GitHub-specific
    └── workflows/
        └── tests.yml                  # Auto-run tests (optional)
```

---

## File-by-File Breakdown

### 🔴 CRITICAL: README.md (Project Landing Page)

```markdown
# Gold Price Forecasting: ML/DL Comparison

**A capstone project comparing baseline statistical models, machine learning, 
and deep learning approaches for predicting short-term gold price movements.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Quick Results

| Model | RMSE | MAE | Directional Accuracy |
|-------|------|-----|----------------------|
| **Random Forest** | **0.00890** | **0.00640** | **50.81%** |
| SMA(20) | 0.00891 | 0.00643 | 50.46% |
| XGBoost | 0.00929 | 0.00687 | 47.48% |
| LSTM | 0.00937 | 0.00691 | 49.66% |
| GRU | 0.00960 | 0.00709 | 51.11% |
| Naive | 0.01249 | 0.00945 | 44.97% |
| MLP | 0.02699 | 0.02302 | 49.70% |

## 🎯 Key Finding
**Random Forest performs best**, nearly matching a simple SMA baseline. 
Directional accuracy ~50% for all models (hard to predict direction).
[See full results →](docs/RESULTS.md)

## 🚀 Quick Start

### Run Everything (Automated)
```bash
# Install dependencies
pip install -r requirements.txt

# Run all notebooks automatically
python scripts/run_all_notebooks.py
```

**Results saved to:**
- Metrics: `results/metrics/walk_forward_results.csv`
- Figures: `results/figures/`

### Run Specific Notebook
```bash
jupyter notebook notebooks/08_walk_forward_validation.ipynb
```

## 📚 Documentation

- **[📖 Full Methodology](docs/METHODOLOGY.md)** - Data sources, features, models
- **[📊 Results Summary](docs/RESULTS.md)** - Detailed metrics and analysis
- **[💡 Conclusions](docs/CONCLUSIONS.md)** - Insights and recommendations
- **[📓 Notebooks Guide](notebooks/README.md)** - How to read the analysis

## 🏗️ Project Structure

```
notebooks/        - Exploratory analysis (8 notebooks)
docs/             - Documentation + figures
data/             - Data sources (raw → processed)
results/          - Model outputs & metrics
scripts/          - Automation scripts
```

## 📖 How to Reproduce

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run everything automatically**
   ```bash
   python scripts/run_all_notebooks.py
   ```
   
   Or run notebooks individually:
   ```bash
   jupyter notebook notebooks/01_data_collection.ipynb
   jupyter notebook notebooks/02_data_cleaning.ipynb
   # ... continue for each notebook
   ```

3. **View results**
   - CSV metrics: `results/metrics/walk_forward_results.csv`
   - Generated plots: `results/figures/`

## 🔍 Data

**Sources:**
- Gold prices: Yahoo Finance
- EUR/USD: OANDA
- VIX, SPY: Yahoo Finance
- Treasury 10Y, DXY, Oil: [Source]

**Time Period:** 2003-2024 (21 years, daily data)

**Final Dataset:** 5,469 rows × 43 columns
- 1 target variable (next-day gold return)
- 42 features (lagged returns, technicals, macro indicators)

[Data README →](data/raw/README.md)

## 🛠️ Technical Stack

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Deep Learning:** tensorflow, keras
- **Visualization:** matplotlib, seaborn
- **Time Series:** statsmodels

**Python Version:** 3.8+

## 📊 Models Evaluated

### Baselines
- Naive (yesterday's return)
- SMA(20) (20-day moving average)
- ARIMA(1,0,1) (statistical model)

### Machine Learning
- Random Forest (300 trees, max_depth=6)
- XGBoost (500 trees, learning_rate=0.05)

### Deep Learning
- MLP (64→32→1 dense layers)
- LSTM (32 units, 5-timestep sequences)
- GRU (32 units, 5-timestep sequences)

## 🔬 Methodology Highlights

- **Train-Test Split:** 2003-2015 training, 2016-2024 testing
- **Validation:** Walk-forward validation (9 folds)
- **No Data Leakage:** Lagged features from feature engineering
- **Metrics:** RMSE, MAE, Directional Accuracy

[Full Methodology →](docs/METHODOLOGY.md)

## 📈 Results Highlights

### Performance Ranking (Average 2016-2024)

**RMSE (Lower is Better):**
1. Random Forest: 0.00890
2. SMA(20): 0.00891
3. XGBoost: 0.00929

**Directional Accuracy (Higher is Better):**
1. GRU: 51.11%
2. Random Forest: 50.81%
3. SMA(20): 50.46%

### Key Insights
- Simple baselines beat complex models (gold follows mean-reversion)
- Predicting direction is harder than magnitude
- Deep learning struggles without more training (epochs=20 likely too low)
- 2020 volatility (COVID) particularly challenging

[Detailed Results →](docs/RESULTS.md)

## 💡 Conclusions & Recommendations

1. **Use Random Forest for production** - Best RMSE, robust across years
2. **Consider ensemble approach** - RF + SMA combination for stability
3. **Directional prediction difficult** - Focus on magnitude for better results
4. **Future improvements:**
   - Increase neural network epochs (50+)
   - Add attention mechanisms
   - Test ensemble methods
   - Include sentiment/news data

[Full Conclusions →](docs/CONCLUSIONS.md)

## 📝 Author

**[Your Name]**
- LinkedIn: [Your Profile]
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MSCF Program at [University]
- Advisors: [Professor Name]
- Data sources: Yahoo Finance, OANDA, etc.

---

**Last Updated:** December 2025
**Status:** Complete ✅
```

---

### 🟢 IMPORTANT: docs/README.md

```markdown
# Documentation

Welcome to the documentation! Start here if you want to understand what was done.

## Reading Order

1. **[PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md)** - What problem are we solving? (5 min)
2. **[METHODOLOGY.md](METHODOLOGY.md)** - How did we do it? (10 min)
3. **[RESULTS.md](RESULTS.md)** - What did we find? (10 min)
4. **[CONCLUSIONS.md](CONCLUSIONS.md)** - What does it mean? (5 min)

## Quick Answers

- **"What's this project about?"** → [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md)
- **"What models did you use?"** → [METHODOLOGY.md](METHODOLOGY.md)
- **"Which model won?"** → [RESULTS.md](RESULTS.md)
- **"What should we do now?"** → [CONCLUSIONS.md](CONCLUSIONS.md)
- **"Show me pictures"** → [figures/](figures/)
```

---

### 📕 IMPORTANT: docs/METHODOLOGY.md

```markdown
# Methodology

## Data Collection

**Sources:**
- Gold prices: Yahoo Finance
- Macro indicators: [sources]
- Time period: 2003-2024

**Variables:**
- 7 time series merged
- 5,732 initial rows
- Forward-filled missing values
- Final: 5,469 rows (99.5% retention)

## Feature Engineering

**Created 42 features:**
- Lagged returns (5)
- Volatility (3)
- Moving averages (3)
- Momentum (3)
- Technical indicators (RSI, MACD)
- Macro lags (8 features × 3 lags)

**Target:** Next-day gold return (shift(-1) for no lookahead bias)

## Models

### Baselines
- **Naive:** Yesterday's return
- **SMA(20):** Static 20-day moving average
- **ARIMA(1,0,1):** Statistical model

### Machine Learning
- **Random Forest:** 300 trees, max_depth=6
- **XGBoost:** 500 trees, learning_rate=0.05

### Deep Learning
- **MLP:** 64→32→1 dense layers, 50 epochs
- **LSTM:** 32 units, 5-timestep sequences, 40 epochs
- **GRU:** 32 units, 5-timestep sequences, 40 epochs

## Validation

**Train-Test Split:**
- Training: 2003-2015 (3,123 samples)
- Testing: 2016-2024 (9 annual folds)

**Metrics:**
- RMSE (penalizes large errors)
- MAE (robust to outliers)
- Directional Accuracy (% correct up/down)

[Full details in notebooks/08_walk_forward_validation.ipynb]
```

---

### 📗 IMPORTANT: docs/RESULTS.md

```markdown
# Results

## Summary Table (Average 2016-2024)

| Model | RMSE ↓ | MAE ↓ | Dir.Acc ↑ |
|-------|--------|-------|-----------|
| Random Forest | 0.00890 | 0.00640 | 50.81% |
| SMA(20) | 0.00891 | 0.00643 | 50.46% |
| XGBoost | 0.00929 | 0.00687 | 47.48% |
| LSTM | 0.00937 | 0.00691 | 49.66% |
| GRU | 0.00960 | 0.00709 | 51.11% |
| Naive | 0.01249 | 0.00945 | 44.97% |
| MLP | 0.02699 | 0.02302 | 49.70% |

## Visualizations

### RMSE by Year
[Image: rmse_by_year.png]

### Directional Accuracy by Year
[Image: directional_accuracy.png]

### Model Rankings
[Image: model_rankings.png]

## Detailed Findings

1. **Random Forest wins** - Lowest RMSE (0.00890)
2. **SMA surprisingly close** - Nearly matches RF
3. **Direction hard to predict** - All ~50% (random guess is 50%)
4. **MLP struggles** - Likely needs more training
5. **2020 anomaly** - COVID volatility challenges all models

[Full analysis: notebooks/08_walk_forward_validation.ipynb]
```

---

### 📙 IMPORTANT: docs/CONCLUSIONS.md

```markdown
# Conclusions & Recommendations

## Key Findings

1. **Random Forest is best** for RMSE (magnitude prediction)
2. **GRU is best** for directional accuracy
3. **SMA baseline is competitive** - Gold has mean-reversion patterns
4. **Direction prediction is hard** - Most models ~50% accuracy
5. **Volatility periods tough** - 2020 COVID challenging for DL

## Recommendations

### For Production
- Use Random Forest model
- Update daily with walk-forward window
- Monitor 2020-like volatile periods

### For Future Work
- Increase neural network epochs (50+ instead of 20)
- Add attention mechanisms (Transformer)
- Ensemble approach (RF + GRU)
- Include sentiment/news data

## Limitations

- Daily data only (may miss intraday patterns)
- Historical data (2003-2024) may not reflect future
- No transaction costs considered
- Simple models may be near efficient market hypothesis

## Next Steps

1. Deploy RF model to production
2. Create retraining pipeline
3. Monitor performance over time
4. Consider ensemble methods
```

---

### 📓 IMPORTANT: notebooks/README.md

```markdown
# Notebooks Guide

## Reading Order

Run notebooks in numbered order (they depend on each other):

1. **00_project_setup.ipynb** - Setup environment, verify packages
2. **01_data_collection.ipynb** - Load raw data from sources
3. **02_data_cleaning.ipynb** - Handle missing values, outliers
4. **03_eda.ipynb** - Exploratory analysis, distributions
5. **04_feature_engineering.ipynb** - Create 42 features
6. **05_baseline_models.ipynb** - Train 3 baseline models on 2016
7. **06_ml_models.ipynb** - Train RF & XGB on 2016
8. **07_deep_learning.ipynb** - Train MLP, LSTM, GRU on 2016
9. **08_walk_forward_validation.ipynb** - 9-year walk-forward test
10. **09_final_summary.ipynb** - Synthesize all results

## How to Run

### Option 1: Run All Automatically
```bash
python scripts/run_all_notebooks.py
```

### Option 2: Run One at a Time
```bash
jupyter notebook notebooks/01_data_collection.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb
# etc...
```

## What Each Notebook Does

| Notebook | Purpose | Time |
|----------|---------|------|
| 00_setup | Verify Python, packages installed | 1 min |
| 01_collection | Download & load 7 data sources | 2 min |
| 02_cleaning | Forward-fill NaNs, final dataset | 2 min |
| 03_eda | Distributions, correlations, plots | 5 min |
| 04_features | Engineer 42 features | 3 min |
| 05_baselines | Simple models on 2016 | 2 min |
| 06_ml | RF & XGB on 2016 | 5 min |
| 07_dl | MLP, LSTM, GRU on 2016 | 10 min |
| 08_validation | Walk-forward 2016-2024 | 30 min |
| 09_summary | Final tables & visualizations | 5 min |

## File Outputs

Each notebook saves outputs for next notebook:

```
01 → data/processed/cleaned_data.csv
02 → data/processed/cleaned_data.csv
03 → (plots only)
04 → data/processed/modeling_dataset.csv
05 → results/baseline_results.csv
06 → results/ml_results.csv
07 → results/dl_results.csv
08 → results/walk_forward_results.csv
09 → docs/RESULTS.md + figures/
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- pandas, numpy
- scikit-learn, xgboost
- tensorflow, keras
- matplotlib, seaborn
- statsmodels
```

---

### 🚀 CRITICAL: scripts/run_all_notebooks.py

```python
#!/usr/bin/env python
"""
Automated pipeline to run all notebooks in sequence
Usage: python scripts/run_all_notebooks.py
"""

import subprocess
import sys
from pathlib import Path

def run_notebooks():
    notebooks = [
        'notebooks/00_project_setup.ipynb',
        'notebooks/01_data_collection.ipynb',
        'notebooks/02_data_cleaning.ipynb',
        'notebooks/03_eda.ipynb',
        'notebooks/04_feature_engineering.ipynb',
        'notebooks/05_baseline_models.ipynb',
        'notebooks/06_ml_models.ipynb',
        'notebooks/07_deep_learning.ipynb',
        'notebooks/08_walk_forward_validation.ipynb',
        'notebooks/09_final_summary.ipynb',
    ]
    
    print("=" * 80)
    print("🚀 GOLD PRICE FORECASTING CAPSTONE - AUTOMATED PIPELINE")
    print("=" * 80)
    print(f"\n📋 Will run {len(notebooks)} notebooks...\n")
    
    failed = []
    
    for i, nb in enumerate(notebooks, 1):
        nb_name = Path(nb).stem
        print(f"\n[{i}/{len(notebooks)}] Running {nb_name}...")
        print("-" * 80)
        
        try:
            # Try using papermill (if installed)
            try:
                import papermill as pm
                pm.execute_notebook(nb, nb, kernel_name='python3')
            except ImportError:
                # Fallback to jupyter nbconvert
                subprocess.run(
                    ['jupyter', 'nbconvert', '--to', 'notebook', 
                     '--execute', '--inplace', '--ExecutePreprocessor.timeout=600',
                     nb],
                    check=True,
                    capture_output=False
                )
            
            print(f"✅ {nb_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {nb_name} FAILED: {str(e)}")
            failed.append((nb_name, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PIPELINE SUMMARY")
    print("=" * 80)
    
    if not failed:
        print(f"✅ ALL {len(notebooks)} NOTEBOOKS COMPLETED SUCCESSFULLY!")
        print("\n📁 Results saved to:")
        print("   - Metrics: results/metrics/walk_forward_results.csv")
        print("   - Figures: results/figures/")
        print("   - Analysis: docs/RESULTS.md")
        print("\n🎉 Ready for review!")
    else:
        print(f"\n⚠️  {len(failed)} notebook(s) failed:\n")
        for nb_name, error in failed:
            print(f"   ❌ {nb_name}")
            print(f"      Error: {error[:100]}...")
        sys.exit(1)

if __name__ == '__main__':
    try:
        run_notebooks()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
```

---

### 📄 .gitignore

```
# Data (large files)
data/raw/
data/processed/
!data/.gitkeep
!data/raw/.gitkeep
!data/raw/README.md

# Results (generated)
results/
!results/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment
.env
.conda/

# OSX
.DS_Store

# Jupyter
.ipynb_checkpoints
```

---

### requirements.txt

```
# Data Processing
pandas==1.5.3
numpy==1.24.0

# Machine Learning
scikit-learn==1.2.0
xgboost==1.7.0
statsmodels==0.13.5

# Deep Learning
tensorflow==2.12.0
keras==2.12.0

# Visualization
matplotlib==3.7.0
seaborn==0.12.2
plotly==5.13.0

# Automation
papermill==2.4.0
jupyter==1.0.0
jupyterlab==3.5.3

# Utilities
pyyaml==6.0
python-dotenv==1.0.0
```

---

### LICENSE (MIT)

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## GitHub Best Practices

### Repository Settings

1. **Description:** 
   ```
   ML/DL models comparing approaches to gold price forecasting 
   (Capstone project)
   ```

2. **Topics:** 
   ```
   machine-learning deep-learning time-series capstone gold-price-forecasting
   ```

3. **Visibility:** Public

4. **License:** MIT (or your choice)

### README Badges (Add to top of README.md)

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]
(https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/yourusername/gold-capstone/blob/main/notebooks/00_project_setup.ipynb)
```

### GitHub Pages (Optional - Showcase Documentation)

Create `.github/workflows/deploy.yml` to auto-publish docs.

---

## File Checklist

Before uploading to GitHub:

- [ ] README.md is comprehensive (500+ words)
- [ ] docs/METHODOLOGY.md complete
- [ ] docs/RESULTS.md complete
- [ ] docs/CONCLUSIONS.md complete
- [ ] All notebooks numbered (00-09)
- [ ] notebooks/README.md explains order
- [ ] scripts/run_all_notebooks.py works
- [ ] requirements.txt updated
- [ ] .gitignore excludes data/results/
- [ ] LICENSE file present
- [ ] No secrets/credentials in code
- [ ] All notebooks run without errors
- [ ] Figures saved to docs/figures/
- [ ] No large files committed (< 100MB total)

---

## Showcase Your Project

### In README, highlight:
1. **Problem** - What are you solving? (1 sentence)
2. **Approach** - How? (3 features)
3. **Results** - Key metrics in table
4. **Quick Start** - Run in 2 commands
5. **Documentation** - Links to deep dives

### On GitHub Profile:
- Pin this repository
- Add to portfolio
- Share on LinkedIn
- Link in resume

---

## TL;DR - Minimal Checklist

✅ **Must Have:**
1. Comprehensive README.md
2. docs/ folder with 4 markdown files
3. 10 notebooks (00-09)
4. scripts/run_all_notebooks.py
5. requirements.txt
6. .gitignore
7. LICENSE

✅ **Nice to Have:**
8. Badges in README
9. figures/ folder
10. config/params.yaml

✅ **Final Check:**
11. Everything works end-to-end
12. Clear, professional presentation
13. No sensitive data committed

---

**This structure will impress professors, employers, and GitHub visitors!** 🎯
