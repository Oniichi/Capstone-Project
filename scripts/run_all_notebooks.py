#!/usr/bin/env python3
"""
Run all notebooks in the capstone project sequentially.

This script uses papermill to execute all notebooks in order and track progress.

Usage:
    python run_all_notebooks.py
    
    Options:
    --output-dir DIR    Save executed notebooks to DIR (default: ./executed_notebooks/)
    --skip-errors       Continue if a notebook fails (default: stops on error)
    --timeout SECONDS   Timeout per notebook in seconds (default: 3600)
    --help              Show this help message

Requirements:
    pip install papermill pandas
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def get_notebook_list():
    """Return list of notebooks to run in order."""
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    notebooks = [
        "01_data_collection.ipynb",
        "02_data_cleaning.ipynb",
        "03_exploratory_analysis.ipynb",
        "04_feature_engineering.ipynb",
        "05_baseline_models.ipynb",
        "06_ml_models.ipynb",
        "07_deep_learning.ipynb",
        "08_walk_forward_validation.ipynb",
    ]
    
    return [str(notebooks_dir / nb) for nb in notebooks]


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import papermill
        print("✓ papermill found")
    except ImportError:
        print("✗ papermill not found")
        print("  Install with: pip install papermill")
        return False
    
    try:
        import pandas
        print("✓ pandas found")
    except ImportError:
        print("✗ pandas not found")
        print("  Install with: pip install pandas")
        return False
    
    return True


def run_notebook(notebook_path, output_path=None, timeout=3600):
    """
    Run a single notebook using papermill.
    
    Args:
        notebook_path: Path to input notebook
        output_path: Path to save executed notebook (optional)
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    import papermill as pm
    
    notebook_name = Path(notebook_path).name
    
    try:
        print(f"\n{'='*70}")
        print(f"Running: {notebook_name}")
        print(f"{'='*70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if output_path:
            print(f"Output: {output_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = notebook_path
        
        # Run notebook with papermill
        pm.execute_notebook(
            notebook_path,
            output_path,
            timeout=timeout,
            progress_bar=True,
            log_output=True,
        )
        
        print(f"✓ {notebook_name} completed successfully")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except pm.PapermillExecutionError as e:
        print(f"✗ {notebook_name} failed with execution error:")
        print(f"  {str(e)}")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ {notebook_name} timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"✗ {notebook_name} failed with error:")
        print(f"  {str(e)}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run all capstone notebooks sequentially"
    )
    parser.add_argument(
        "--output-dir",
        default="./executed_notebooks/",
        help="Directory to save executed notebooks (default: ./executed_notebooks/)"
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue if a notebook fails (default: stops on error)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per notebook in seconds (default: 3600)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CAPSTONE PROJECT: AUTOMATIC NOTEBOOK RUNNER")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skip errors: {args.skip_errors}")
    print(f"Timeout per notebook: {args.timeout} seconds")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\n✗ Missing required packages. Install with:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Get notebook list
    notebooks = get_notebook_list()
    print(f"\nNotebooks to run ({len(notebooks)} total):")
    for i, nb in enumerate(notebooks, 1):
        print(f"  {i}. {Path(nb).name}")
    
    # Run notebooks
    output_dir = Path(args.output_dir)
    results = {}
    failed_notebooks = []
    
    for i, notebook_path in enumerate(notebooks, 1):
        notebook_name = Path(notebook_path).name
        
        # Generate output path
        if args.output_dir:
            output_path = output_dir / notebook_name
        else:
            output_path = None
        
        # Run notebook
        success = run_notebook(notebook_path, output_path, args.timeout)
        results[notebook_name] = success
        
        if not success:
            failed_notebooks.append(notebook_name)
            if not args.skip_errors:
                print(f"\n✗ Stopping due to failed notebook (use --skip-errors to continue)")
                break
        else:
            # Print progress
            completed = sum(1 for v in results.values() if v)
            print(f"\nProgress: {completed}/{len(notebooks)} notebooks completed")
    
    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    completed = sum(1 for v in results.values() if v)
    print(f"\nResults: {completed}/{len(notebooks)} notebooks completed")
    
    if failed_notebooks:
        print(f"\n✗ Failed notebooks ({len(failed_notebooks)}):")
        for nb in failed_notebooks:
            print(f"  - {nb}")
        print(f"\nTo debug, run the notebook manually:")
        print(f"  jupyter notebook <notebook_path>")
        return 1
    else:
        print("\n✓ All notebooks completed successfully!")
        print(f"\nOutputs saved to: {args.output_dir}")
        print(f"\nKey outputs:")
        print(f"  - data/processed/cleaned_data.csv (from 02)")
        print(f"  - data/processed/modeling_dataset.csv (from 04)")
        print(f"  - results/metrics/walk_forward_results.csv (from 08)")
        print(f"  - results/figures/*.png (from 05-08)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
