"""
SIMPLE USAGE GUIDE FOR PERSISTENT MODEL OPTIMIZER
==================================================

This system solves the problem of stochastic model training causing
different results between optimization and backtesting.

KEY INSIGHT: Train models ONCE, save them, reuse them forever.
"""

import sys
import os

# IMPORTANT: Adjust these paths to match YOUR setup
# Path to where you placed optimize_with_model_persistence.py
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))  # Current directory
sys.path.insert(0, PROJECT_PATH)

# If your modules are in a different location, uncomment and adjust:
# sys.path.insert(0, '/path/to/your/project')  # e.g., 'C:/Users/YourName/algs'

from optimize_with_model_persistence import (
    PersistentModelOptimizer,
    PersistentModelBacktester
)

# =============================================================================
# STEP 1: ONE-TIME OPTIMIZATION
# =============================================================================

def step1_optimize_once(data_file):
    """
    Run this ONCE to find optimal parameters and save trained models
    
    This will:
    1. Train DMD and SINDy models (one time, deterministic)
    2. Test many strategy parameter combinations
    3. Save the best models and parameters
    
    Output files:
    - optimized_models.pkl (the trained models)
    - optimization_results.json (performance metrics)
    """
    
    print("="*80)
    print("STEP 1: ONE-TIME OPTIMIZATION")
    print("="*80)
    print("This will take a while but only needs to be done ONCE")
    print("="*80)
    
    optimizer = PersistentModelOptimizer(
        data_file=data_file,
        target_metric='log_price_ratio_sma20'
    )
    
    # Optimize with your desired parameter ranges
    best_result = optimizer.optimize_and_save_models(
        # Parameter ranges to test
        prediction_horizons=[5, 7, 10, 14],
        entry_thresholds=[0.5, 1.0, 1.5, 2.0],
        take_profit_pcts=[3.0, 5.0, 7.0, 10.0],
        stop_loss_pcts=[-2.0, -3.0, -5.0],
        max_holding_days_list=[7, 10, 14, 20],
        profit_windows=[3, 5, 7],
        
        # Number of random combinations to test
        n_iterations=100,  # Increase for better results (but slower)
        
        # Where to save the results
        model_save_path='optimized_models.pkl',
        results_save_path='optimization_results.json'
    )
    
    print("\n✅ STEP 1 COMPLETE!")
    print("Models saved to: optimized_models.pkl")
    print("Results saved to: optimization_results.json")
    print("\nYou can now run STEP 2 as many times as you want!")
    
    return best_result


# =============================================================================
# STEP 2: BACKTEST WITH SAVED MODELS (RUN MANY TIMES)
# =============================================================================

def step2_backtest_with_saved_models(data_file, model_file='optimized_models.pkl'):
    """
    Run this MANY TIMES to backtest with the saved models
    
    This will:
    1. Load the pre-trained models
    2. Load the optimal parameters
    3. Run backtest
    4. GUARANTEE same results as optimization!
    
    You can run this on:
    - The same data file (to verify it matches optimization)
    - Different data files (to test on new stocks)
    - Different time periods (walk-forward testing)
    """
    
    print("="*80)
    print("STEP 2: BACKTEST WITH SAVED MODELS")
    print("="*80)
    
    backtester = PersistentModelBacktester(model_path=model_file)
    
    # This will automatically load the models and parameters
    results = backtester.backtest(
        data_file=data_file,
        detailed_output=True
    )
    
    print("\n✅ STEP 2 COMPLETE!")
    print("Results should EXACTLY match optimization if same data file used")
    
    return results


# =============================================================================
# STEP 3 (OPTIONAL): VERIFY REPRODUCIBILITY
# =============================================================================

def step3_verify_reproducibility(data_file):
    """
    Run this to verify that the backtest gives EXACTLY the same results
    as the optimization
    
    This proves the system works correctly!
    """
    
    print("="*80)
    print("STEP 3: VERIFY REPRODUCIBILITY")
    print("="*80)
    
    # Load the optimization results
    import json
    with open('optimization_results.json', 'r') as f:
        opt_results = json.load(f)
    
    print("\n📊 OPTIMIZATION RESULTS (from Step 1):")
    print(f"Total Return: {opt_results['best_performance']['total_return']:.2%}")
    print(f"Sharpe Ratio: {opt_results['best_performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {opt_results['best_performance']['max_drawdown']:.2%}")
    print(f"Win Rate: {opt_results['best_performance']['win_rate']:.2%}")
    print(f"Trades: {opt_results['best_performance']['num_trades']}")
    
    # Run backtest
    backtester = PersistentModelBacktester(model_path='optimized_models.pkl')
    backtest_results = backtester.backtest(data_file, detailed_output=False)
    
    print("\n📊 BACKTEST RESULTS (from Step 2):")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"Win Rate: {backtest_results['win_rate']:.2%}")
    print(f"Trades: {backtest_results['num_trades']}")
    
    # Check if they match
    tolerance = 0.001  # 0.1% tolerance for floating point errors
    
    matches = (
        abs(backtest_results['total_return'] - opt_results['best_performance']['total_return']) < tolerance and
        abs(backtest_results['sharpe_ratio'] - opt_results['best_performance']['sharpe_ratio']) < tolerance and
        abs(backtest_results['max_drawdown'] - opt_results['best_performance']['max_drawdown']) < tolerance and
        backtest_results['num_trades'] == opt_results['best_performance']['num_trades']
    )
    
    print("\n" + "="*80)
    if matches:
        print("✅ SUCCESS! Results match exactly!")
        print("The system is working correctly - same models = same results")
    else:
        print("⚠️ WARNING: Results don't match!")
        print("This shouldn't happen - there may be a bug")
    print("="*80)
    
    return matches


# =============================================================================
# COMPLETE WORKFLOW EXAMPLE
# =============================================================================

def complete_workflow_example():
    """
    Complete example showing all steps
    """
    
    DATA_FILE = 'your_stock_data.csv'  # ← CHANGE THIS TO YOUR FILE
    
    print("\n🚀 STARTING COMPLETE WORKFLOW")
    print("="*80)
    
    # STEP 1: Optimize (do once)
    print("\n--- Running Step 1: Optimization ---")
    step1_optimize_once(DATA_FILE)
    
    # STEP 2: Backtest (do many times)
    print("\n--- Running Step 2: Backtest ---")
    step2_backtest_with_saved_models(DATA_FILE)
    
    # STEP 3: Verify (optional but recommended)
    print("\n--- Running Step 3: Verification ---")
    matches = step3_verify_reproducibility(DATA_FILE)
    
    if matches:
        print("\n🎉 COMPLETE SUCCESS!")
        print("You now have:")
        print("1. Optimized models saved in optimized_models.pkl")
        print("2. Optimal parameters saved in optimization_results.json")
        print("3. Verified that backtest = optimization")
        print("\nYou can now use these models for live trading!")
    
    return matches


# =============================================================================
# QUICK START GUIDE
# =============================================================================

"""
QUICK START GUIDE
=================

1. First time (optimization):
   
   from simple_usage_guide import step1_optimize_once
   step1_optimize_once('your_data.csv')
   
   This creates:
   - optimized_models.pkl
   - optimization_results.json

2. Every other time (backtesting):
   
   from simple_usage_guide import step2_backtest_with_saved_models
   step2_backtest_with_saved_models('your_data.csv')
   
   OR test on different data:
   
   step2_backtest_with_saved_models('different_stock.csv')

3. Verify it works (optional):
   
   from simple_usage_guide import step3_verify_reproducibility
   step3_verify_reproducibility('your_data.csv')

THAT'S IT! No more stochastic differences!
"""


if __name__ == "__main__":
    """
    Example execution
    """
    
    # Replace with your actual data file
    DATA_FILE = 'SPY_data.csv'  # Example
    
    # If you want to run the complete workflow:
    # complete_workflow_example()
    
    # Or run steps individually:
    
    # Step 1 (run once):
    # step1_optimize_once(DATA_FILE)
    
    # Step 2 (run many times):
    # step2_backtest_with_saved_models(DATA_FILE)
    
    # Step 3 (verify):
    # step3_verify_reproducibility(DATA_FILE)
    
    print("\n📚 See the docstrings above for usage instructions")
    print("Uncomment the function you want to run")
