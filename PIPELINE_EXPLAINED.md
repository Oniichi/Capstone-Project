# Understanding "Pipeline" in Machine Learning

## What is a Pipeline?

A **pipeline** is simply an **automated workflow** that chains multiple steps together.

Instead of running things manually one-by-one:
```python
# Manual (BAD)
data = load_data()
data = clean_data(data)
data = engineer_features(data)
model = train_model(data)
results = evaluate_model(model, data)
save_results(results)
```

A **pipeline orchestrates** all these steps automatically:
```python
# Automated Pipeline (GOOD)
pipeline = GoldForecastingPipeline()
results = pipeline.run()  # One command does everything
```

---

## Simple Example: Your Gold Price Project

### **WITHOUT Pipeline (What you're doing now)**

Your notebooks are like manual steps:
```
Notebook 01: "I'll load data manually"
↓ (You run it, wait, look at output)
Notebook 02: "Now I'll clean it manually"
↓ (You run it, wait, look at output)
Notebook 03: "Now I'll engineer features manually"
↓ (You run it, wait, look at output)
... repeat 8 times ...
Notebook 08: "Finally, results!"
```

**Problem**: If you want to re-run everything with different hyperparameters, you have to manually run all 8 notebooks again.

---

### **WITH Pipeline (Automation)**

```python
# src/pipeline/orchestrator.py

class GoldForecastingPipeline:
    def __init__(self, config_path='config/default.yaml'):
        self.config = load_config(config_path)
        self.data = None
        self.results = {}
    
    def run(self):
        """Execute the entire pipeline in sequence"""
        print("🚀 Starting pipeline...")
        
        self.step_1_load_data()      # Replaces Notebook 01
        self.step_2_clean_data()     # Replaces Notebook 02
        self.step_3_explore_data()   # Replaces Notebook 03
        self.step_4_engineer_features() # Replaces Notebook 04
        self.step_5_train_baselines()   # Replaces Notebook 05
        self.step_6_train_ml_models()   # Replaces Notebook 06
        self.step_7_train_dl_models()   # Replaces Notebook 07
        self.step_8_walk_forward_validation() # Replaces Notebook 08
        self.step_9_generate_report()   # Replaces Notebook 09
        
        print("✅ Pipeline complete!")
        return self.results
    
    def step_1_load_data(self):
        """Load raw CSV files and merge"""
        print("\n[1/9] Loading data...")
        self.data = load_raw_data(
            data_dir=self.config['data']['raw_dir']
        )
        print(f"   ✓ Loaded {len(self.data)} rows")
    
    def step_2_clean_data(self):
        """Handle missing values, outliers"""
        print("\n[2/9] Cleaning data...")
        self.data = clean_missing_values(self.data)
        self.data = remove_outliers(self.data)
        print(f"   ✓ Cleaned data shape: {self.data.shape}")
    
    def step_3_explore_data(self):
        """Generate EDA visualizations"""
        print("\n[3/9] Exploring data...")
        plots = generate_eda_plots(self.data)
        save_plots(plots, output_dir='docs/figures/')
        print(f"   ✓ Generated {len(plots)} plots")
    
    def step_4_engineer_features(self):
        """Create lagged features, technical indicators"""
        print("\n[4/9] Engineering features...")
        self.data = add_lagged_features(self.data, lags=[1,2,3,4,5])
        self.data = add_volatility_features(self.data)
        self.data = add_technical_indicators(self.data)
        print(f"   ✓ Final dataset: {self.data.shape}")
        self.data.to_csv('data/processed/modeling_dataset.csv')
    
    def step_5_train_baselines(self):
        """Train Naive, SMA, ARIMA on 2016 data"""
        print("\n[5/9] Training baseline models...")
        train, test = split_by_year(self.data, 2016)
        
        baselines = {
            'naive': NaiveForecaster().fit(train).predict(test),
            'sma20': SMAForecaster(window=20).fit(train).predict(test),
            'arima': ARIMAForecaster().fit(train).predict(test)
        }
        
        self.results['baselines'] = {
            name: compute_metrics(test['target'], pred)
            for name, pred in baselines.items()
        }
        print(f"   ✓ Trained {len(baselines)} baseline models")
    
    def step_6_train_ml_models(self):
        """Train Random Forest and XGBoost on 2016 data"""
        print("\n[6/9] Training ML models...")
        train, test = split_by_year(self.data, 2016)
        
        ml_models = {
            'rf': RandomForestRegressor(**self.config['models']['rf']),
            'xgb': XGBRegressor(**self.config['models']['xgb'])
        }
        
        self.results['ml'] = {}
        for name, model in ml_models.items():
            model.fit(train.drop('target', axis=1), train['target'])
            pred = model.predict(test.drop('target', axis=1))
            self.results['ml'][name] = compute_metrics(test['target'], pred)
        
        print(f"   ✓ Trained {len(ml_models)} ML models")
    
    def step_7_train_dl_models(self):
        """Train MLP, LSTM, GRU on 2016 data"""
        print("\n[7/9] Training DL models...")
        train, test = split_by_year(self.data, 2016)
        
        dl_models = {
            'mlp': build_mlp(input_dim=train.shape[1]-1),
            'lstm': build_lstm(input_dim=train.shape[1]-1),
            'gru': build_gru(input_dim=train.shape[1]-1)
        }
        
        self.results['dl'] = {}
        for name, model in dl_models.items():
            model.fit(train.drop('target', axis=1), train['target'],
                     epochs=self.config['models'][name]['epochs'],
                     verbose=0)
            pred = model.predict(test.drop('target', axis=1))
            self.results['dl'][name] = compute_metrics(test['target'], pred)
        
        print(f"   ✓ Trained {len(dl_models)} DL models")
    
    def step_8_walk_forward_validation(self):
        """Run walk-forward validation 2016-2024"""
        print("\n[8/9] Running walk-forward validation...")
        wfv_results = []
        
        for year in range(2016, 2025):
            train = self.data[self.data.index < f'{year}-01-01']
            test = self.data[(self.data.index >= f'{year}-01-01') & 
                            (self.data.index <= f'{year}-12-31')]
            
            year_metrics = {}
            
            # Train all 7 models
            for model_type in ['baseline', 'ml', 'dl']:
                for model_name in self.get_models_by_type(model_type):
                    model = train_model(model_name, train)
                    pred = model.predict(test)
                    year_metrics[f'{model_name}'] = compute_metrics(
                        test['target'], pred
                    )
            
            wfv_results.append({'year': year, **year_metrics})
        
        self.results['walk_forward'] = pd.DataFrame(wfv_results)
        self.results['walk_forward'].to_csv(
            'results/metrics/walk_forward_results.csv'
        )
        print(f"   ✓ Completed 9-year walk-forward validation")
    
    def step_9_generate_report(self):
        """Create final summary and visualizations"""
        print("\n[9/9] Generating final report...")
        
        # Create comparison table
        summary = create_summary_table(self.results)
        
        # Create visualizations
        plots = [
            plot_rmse_by_year(self.results['walk_forward']),
            plot_directional_accuracy(self.results['walk_forward']),
            plot_model_rankings(summary),
            plot_feature_importance(self.results['feature_importance'])
        ]
        
        # Save report
        save_report(
            summary=summary,
            plots=plots,
            output_path='docs/03_RESULTS.md'
        )
        
        print(f"   ✓ Report saved to docs/03_RESULTS.md")
        return self.results
```

---

## How to Use This Pipeline

### **Option 1: Run Everything**
```bash
python scripts/train.py
```

This single command does:
1. Loads data
2. Cleans it
3. Engineers features
4. Trains 7 models (baselines + ML + DL)
5. Runs walk-forward validation
6. Generates report
7. Saves all results

All automatically, step-by-step, with progress messages.

### **Option 2: Run Just One Step**
```python
from src.pipeline.orchestrator import GoldForecastingPipeline

pipeline = GoldForecastingPipeline()
pipeline.step_1_load_data()
pipeline.step_2_clean_data()
# ... only run what you need
```

### **Option 3: Run with Different Config**
```bash
python scripts/train.py --config config/experimental.yaml
```

Change `config/experimental.yaml`:
```yaml
models:
  rf:
    n_estimators: 500  # More trees
    max_depth: 10      # Deeper
  lstm:
    epochs: 100        # More training
```

Re-run pipeline → Get new results → Compare

---

## Architecture: How Pieces Connect

```
┌─────────────────────────────────────────────┐
│        scripts/train.py                      │
│  (Entry point - user runs this)              │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│  src/pipeline/orchestrator.py                │
│  (Main pipeline that coordinates everything)│
└──────────┬──────────────────────────────────┘
           │
     ┌─────┼─────┬─────────┐
     ↓     ↓     ↓         ↓
   ┌──────────────────┐  ┌──────────────┐
   │ src/data/        │  │ src/models/  │
   │ - loaders.py     │  │ - baselines  │
   │ - cleaners.py    │  │ - ml.py      │
   │ - processors.py  │  │ - dl.py      │
   └──────────────────┘  └──────────────┘
          ↓                       ↓
   ┌──────────────────┐  ┌──────────────────────┐
   │ src/features/    │  │ src/evaluation/      │
   │ - engineering.py │  │ - metrics.py         │
   │ - selectors.py   │  │ - validators.py      │
   └──────────────────┘  └──────────────────────┘
           ↓                       ↓
   ┌──────────────────────────────────────────┐
   │ config/                                   │
   │ - default.yaml (hyperparameters)          │
   │ - paths.yaml (file locations)             │
   └──────────────────────────────────────────┘
           ↓
   ┌──────────────────────────────────────────┐
   │ data/, results/, docs/                    │
   │ (Output files)                            │
   └──────────────────────────────────────────┘
```

---

## Real-World Benefit: Why Pipelines Matter

### **Scenario 1: Debugging**
```
Problem: "Why is 2020 accuracy so bad?"

WITHOUT pipeline:
- Run notebook 1... wait
- Run notebook 2... wait
- ... (8 notebooks total)
- Finally check 2020 results

WITH pipeline:
- Change config: add year filter
- python scripts/train.py --filter-year 2020
- Done in seconds
```

### **Scenario 2: New Requirements**
```
Boss: "Can you add a new model (Gradient Boosting)?"

WITHOUT pipeline:
- Edit notebook 6 manually
- Re-run all 8 notebooks
- Hope nothing breaks
- Cross fingers

WITH pipeline:
- Add to src/models/ml.py: GradientBoostingRegressor()
- Add to config: "gb: {n_estimators: 100}"
- Run: python scripts/train.py
- Automatically trains + evaluates new model
```

### **Scenario 3: Reproducibility**
```
Reviewer: "Can you reproduce your results?"

WITHOUT pipeline:
- Send 8 notebooks
- "Just run them in order..."
- Hope they have same package versions
- Hope they remember what you did

WITH pipeline:
- Send one command: python scripts/train.py
- All results automatically generated
- Fully reproducible
```

---

## Minimal Pipeline Example (For Your Project)

Start **super simple**:

```python
# src/pipeline/simple_pipeline.py

def run_full_pipeline():
    """Execute all steps"""
    
    print("Step 1: Load data")
    df = pd.read_csv('data/processed/modeling_dataset.csv', 
                     index_col=0, parse_dates=True)
    
    print("Step 2: Run walk-forward validation")
    results = []
    for year in range(2016, 2025):
        train = df[df.index < f'{year}-01-01']
        test = df[(df.index >= f'{year}-01-01') & 
                 (df.index <= f'{year}-12-31')]
        
        # Train all models
        models_metrics = train_all_models(train, test)
        results.append({'Year': year, **models_metrics})
    
    print("Step 3: Save results")
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/walk_forward_results.csv')
    
    print("Step 4: Generate report")
    generate_summary_report(results_df)
    
    print("✅ Done!")
    return results_df

# scripts/train.py
if __name__ == '__main__':
    from src.pipeline.simple_pipeline import run_full_pipeline
    run_full_pipeline()
```

Then run:
```bash
python scripts/train.py
```

**That's it!** That's a pipeline.

---

## Summary

| Aspect | Without Pipeline | With Pipeline |
|--------|------------------|---------------|
| **How to run** | Run 8 notebooks manually | `python scripts/train.py` |
| **Time to re-run** | 30 minutes | 2 minutes |
| **Easy to change config?** | Hardcoded in notebooks | Edit YAML, re-run |
| **Reproducible?** | Hope user remembers order | Guaranteed |
| **Add new model?** | Edit notebook, re-run all | Add to src/, auto-integrated |
| **Professional?** | ❌ Messy | ✅ Clean |

---

## Next Step for Your Project

Create `scripts/train.py`:
```python
#!/usr/bin/env python
from src.pipeline.orchestrator import GoldForecastingPipeline

if __name__ == '__main__':
    pipeline = GoldForecastingPipeline()
    pipeline.run()
```

Create `src/pipeline/orchestrator.py`:
- Wrap your notebook logic into functions
- Call them in sequence
- Save results

One command → Everything runs → All results saved automatically.

**That's the pipeline concept!** 🎯
