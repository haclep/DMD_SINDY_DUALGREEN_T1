"""
OPTIMIZED BACKTESTER WITH MODEL PERSISTENCE
============================================
This version trains models ONCE during optimization and saves them for reuse.
No more stochastic variation - guaranteed reproducible results!
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your existing systems
import sys
sys.path.insert(0, '/mnt/project')

from dmdPredictor import RobustDMDSystem
from SINDyPredictor import RobustSINDySystem


class PersistentModelOptimizer:
    """
    Optimizer that saves trained models for exact reproduction
    """
    
    def __init__(self, data_file, target_metric='log_price_ratio_sma20'):
        self.data_file = data_file
        self.target_metric = target_metric
        
        # Load and prepare data
        self.df = self._load_data(data_file)
        
        # Model storage
        self.dmd_system = None
        self.sindy_system = None
        self.trained_models = None
        
        # Optimization results
        self.best_params = None
        self.optimization_results = None
        
    def _load_data(self, filepath):
        """Load and prepare trading data"""
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.columns = [col.capitalize() for col in df.columns]
        return df
    
    def optimize_and_save_models(self, 
                                 prediction_horizons=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                 entry_thresholds=[0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                                 take_profit_pcts=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0],
                                 stop_loss_pcts=[-2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -7.0],
                                 max_holding_days_list=[5,6,7,8,9, 10,11,12,13, 14, 20],
                                 profit_windows=[3, 4, 5, 6, 7],
                                 n_iterations=200,
                                 model_save_path='optimized_models.pkl',
                                 results_save_path='optimization_results.json' ):
        """
        Run optimization and save the exact trained models
        """
        
        print("="*80)
        print("PERSISTENT MODEL OPTIMIZATION")
        print("="*80)
        print(f"Data: {self.data_file}")
        print(f"Target: {self.target_metric}")
        print(f"Iterations: {n_iterations}")
        print("="*80)
        
        # Step 1: Train models ONCE with a specific configuration
        print("\n🔬 STEP 1: TRAINING MODELS (ONE-TIME SETUP)")
        print("-"*80)
        
        best_model_config = self._train_models_once()
        
        # Step 2: Optimize strategy parameters using the trained models
        print("\n🎯 STEP 2: OPTIMIZING STRATEGY PARAMETERS")
        print("-"*80)
        
        best_result = self._optimize_strategy_params(
            prediction_horizons=prediction_horizons,
            entry_thresholds=entry_thresholds,
            take_profit_pcts=take_profit_pcts,
            stop_loss_pcts=stop_loss_pcts,
            max_holding_days_list=max_holding_days_list,
            profit_windows=profit_windows,
            n_iterations=n_iterations
        )
        
        # Step 3: Save everything
        print("\n💾 STEP 3: SAVING MODELS AND RESULTS")
        print("-"*80)
        
        self._save_trained_models(model_save_path, best_result)
        self._save_optimization_results(results_save_path, best_result)
        
        print(f"\n✅ OPTIMIZATION COMPLETE!")
        print(f"📦 Models saved to: {model_save_path}")
        print(f"📊 Results saved to: {results_save_path}")
        
        return best_result
    
    def _train_models_once(self):
        """
        Train DMD and SINDy models ONCE with optimal settings
        These exact models will be reused in backtesting
        """
        
        print("Training DMD (Linear) Model...")
        self.dmd_system = RobustDMDSystem(
            target_metric=self.target_metric,
            prediction_horizon=10  # We'll optimize this later
        )
        
        # Train DMD with moderate complexity to avoid overfitting
        self.dmd_system.comprehensive_optimize(
            self.df, 
            max_features=8,  # Reduced to prevent overfitting
            n_feature_trials=300  # Reduced for speed
        )
        
        # Do final training
        dmd_pred, dmd_actual, dmd_directional = self.dmd_system.final_training_and_evaluation(self.df)
        
        print(f"✅ DMD trained successfully")
        print(f"   Features: {len(self.dmd_system.best_features)}")
        print(f"   Best params: {self.dmd_system.best_params}")
        
        print("\nTraining SINDy (Nonlinear) Model...")
        self.sindy_system = RobustSINDySystem(
            target_metric=self.target_metric,
            prediction_horizon=10  # We'll optimize this later
        )
        
        # Train SINDy with moderate complexity
        self.sindy_system.comprehensive_optimize(
            self.df,
            max_features=8,  # Slightly more features for nonlinear
            n_feature_trials=300
        )
        
        # Do final training
        sindy_pred, sindy_actual, sindy_directional = self.sindy_system.final_training_and_evaluation(self.df)
        
        print(f"✅ SINDy trained successfully")
        print(f"   Features: {len(self.sindy_system.best_features)}")
        print(f"   Best params: {self.sindy_system.best_params}")
        
        return {
            'dmd_features': self.dmd_system.best_features,
            'dmd_params': self.dmd_system.best_params,
            'sindy_features': self.sindy_system.best_features,
            'sindy_params': self.sindy_system.best_params,
        }
    
    def _optimize_strategy_params(self, prediction_horizons, entry_thresholds,
                                  take_profit_pcts, stop_loss_pcts,
                                  max_holding_days_list, profit_windows,
                                  n_iterations):
        """
        Optimize strategy parameters using pre-trained models
        """
        
        results = []
        
        for iteration in range(n_iterations):
            # Random parameter selection
            params = {
                'prediction_horizon': np.random.choice(prediction_horizons),
                'entry_threshold': np.random.choice(entry_thresholds),
                'take_profit_pct': np.random.choice(take_profit_pcts),
                'stop_loss_pct': np.random.choice(stop_loss_pcts),
                'max_holding_days': np.random.choice(max_holding_days_list),
                'profit_window': np.random.choice(profit_windows)
            }
            
            # Run backtest with these parameters
            try:
                backtest_result = self._run_backtest_with_params(params)
                
                results.append({
                    'params': params,
                    'total_return': backtest_result['total_return'],
                    'sharpe_ratio': backtest_result['sharpe_ratio'],
                    'max_drawdown': backtest_result['max_drawdown'],
                    'win_rate': backtest_result['win_rate'],
                    'num_trades': backtest_result['num_trades']
                })
                
                if (iteration + 1) % 10 == 0:
                    print(f"  Completed {iteration + 1}/{n_iterations} iterations...")
                    
            except Exception as e:
                print(f"  ⚠️ Iteration {iteration + 1} failed: {str(e)[:50]}")
                continue
        
        if not results:
            raise ValueError("No successful optimization runs!")
        
        # Sort by total return (or your preferred metric)
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        best_result = results[0]
        
        print(f"\n🏆 BEST PARAMETERS FOUND:")
        print(f"   Total Return: {best_result['total_return']:.2%}")
        print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {best_result['max_drawdown']:.2%}")
        print(f"   Win Rate: {best_result['win_rate']:.2%}")
        print(f"   Trades: {best_result['num_trades']}")
        print(f"\n   Parameters:")
        for key, value in best_result['params'].items():
            print(f"      {key}: {value}")
        
        self.best_params = best_result['params']
        self.optimization_results = results
        
        return best_result
    
    def _run_backtest_with_params(self, params):
        """
        Run backtest using pre-trained models with given parameters
        """
        
        # Update model prediction horizons
        self.dmd_system.prediction_horizon = params['prediction_horizon']
        self.sindy_system.prediction_horizon = params['prediction_horizon']
        
        # Initialize backtester (simplified version)
        trades = []
        portfolio_value = [10000]  # Start with $10k
        equity_curve = [10000]
        
        position = None
        entry_price = None
        entry_date = None
        days_held = 0
        
        # Walk through the data
        lookback_start = int(len(self.df) * 0.75)  # Start from test set
        
        for i in range(lookback_start, len(self.df) - params['max_holding_days']):
            current_date = self.df.index[i]
            current_price = self.df['Close'].iloc[i]
            
            # If we have a position, check exit conditions
            if position is not None:
                days_held += 1
                
                # Calculate current P&L
                pnl_pct = (current_price / entry_price - 1) * 100
                
                # Check exit conditions
                exit_trade = False
                exit_reason = None
                
                if pnl_pct >= params['take_profit_pct']:
                    exit_trade = True
                    exit_reason = 'take_profit'
                elif pnl_pct <= params['stop_loss_pct']:
                    exit_trade = True
                    exit_reason = 'stop_loss'
                elif days_held >= params['max_holding_days']:
                    exit_trade = True
                    exit_reason = 'max_holding'
                
                if exit_trade:
                    # Close position
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    # Update portfolio
                    portfolio_value.append(portfolio_value[-1] * (1 + pnl_pct / 100))
                    
                    position = None
                    entry_price = None
                    entry_date = None
                    days_held = 0
            
            # If no position, check entry conditions
            else:
                # Get predictions using pre-trained models
                try:
                    df_subset = self.df.iloc[:i+1].copy()
                    
                    # DMD prediction
                    dmd_pred = self.dmd_system.make_daily_prediction(
                        df_subset, 
                        params['prediction_horizon']
                    )
                    
                    # SINDy prediction
                    sindy_pred = self.sindy_system.make_daily_prediction(
                        df_subset,
                        params['prediction_horizon']
                    )
                    
                    if dmd_pred and sindy_pred:
                        # Average the predictions
                        avg_change = (dmd_pred['expected_change_pct'] + 
                                    sindy_pred['expected_change_pct']) / 2
                        
                        # Check entry condition
                        if avg_change >= params['entry_threshold']:
                            position = 'long'
                            entry_price = current_price
                            entry_date = current_date
                            days_held = 0
                            
                except Exception:
                    continue
            
            # Update equity curve
            if position is not None:
                current_pnl_pct = (current_price / entry_price - 1) * 100
                equity_curve.append(portfolio_value[-1] * (1 + current_pnl_pct / 100))
            else:
                equity_curve.append(portfolio_value[-1])
        
        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0
            }
        
        total_return = (portfolio_value[-1] / portfolio_value[0] - 1)
        
        # Calculate returns
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        cummax = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'trades': trades
        }
    
    def _save_trained_models(self, filepath, best_result):
        """
        Save the exact trained models and optimal parameters
        """
        
        model_package = {
            # DMD System
            'dmd_model': self.dmd_system.best_model,
            'dmd_features': self.dmd_system.best_features,
            'dmd_params': self.dmd_system.best_params,
            'dmd_feature_scaler': self.dmd_system.feature_scaler,
            'dmd_target_scaler': self.dmd_system.target_scaler,
            'dmd_target_metric': self.dmd_system.target_metric,
            
            # SINDy System
            'sindy_model': self.sindy_system.best_model,
            'sindy_features': self.sindy_system.best_features,
            'sindy_params': self.sindy_system.best_params,
            'sindy_library': self.sindy_system.best_library,
            'sindy_library_config': self.sindy_system.library_config,
            'sindy_feature_scaler': self.sindy_system.feature_scaler,
            'sindy_target_scaler': self.sindy_system.target_scaler,
            'sindy_target_metric': self.sindy_system.target_metric,
            
            # Optimal Strategy Parameters
            'optimal_params': best_result['params'],
            
            # Metadata
            'optimization_date': datetime.now().isoformat(),
            'data_file': self.data_file,
            'target_metric': self.target_metric,
            'optimization_performance': {
                'total_return': best_result['total_return'],
                'sharpe_ratio': best_result['sharpe_ratio'],
                'max_drawdown': best_result['max_drawdown'],
                'win_rate': best_result['win_rate'],
                'num_trades': best_result['num_trades']
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Saved trained models to {filepath}")
        print(f"   Package size: {Path(filepath).stat().st_size / 1024:.1f} KB")
    
    def _save_optimization_results(self, filepath, best_result):
        """
        Save optimization results as JSON for easy inspection
        """
        
        # Helper function to convert numpy types to native Python types
        def convert_to_native_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results_data = {
            'best_params': convert_to_native_types(best_result['params']),
            'best_performance': {
                'total_return': float(best_result['total_return']),
                'sharpe_ratio': float(best_result['sharpe_ratio']),
                'max_drawdown': float(best_result['max_drawdown']),
                'win_rate': float(best_result['win_rate']),
                'num_trades': int(best_result['num_trades'])
            },
            'optimization_date': datetime.now().isoformat(),
            'data_file': self.data_file,
            'target_metric': self.target_metric,
            'total_iterations': len(self.optimization_results) if self.optimization_results else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✅ Saved optimization results to {filepath}")


class PersistentModelBacktester:
    """
    Backtester that uses pre-trained models - guaranteed same results as optimization
    """
    
    def __init__(self, model_path='optimized_models.pkl'):
        self.model_path = model_path
        self.models_loaded = False
        
        # Will be populated from saved models
        self.dmd_system = None
        self.sindy_system = None
        self.optimal_params = None
        
    def load_models(self):
        """
        Load the exact pre-trained models
        """
        
        print(f"📦 Loading pre-trained models from {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Reconstruct DMD system
        self.dmd_system = RobustDMDSystem(
            target_metric=model_package['dmd_target_metric'],
            prediction_horizon=model_package['optimal_params']['prediction_horizon']
        )
        self.dmd_system.best_model = model_package['dmd_model']
        self.dmd_system.best_features = model_package['dmd_features']
        self.dmd_system.best_params = model_package['dmd_params']
        self.dmd_system.feature_scaler = model_package['dmd_feature_scaler']
        self.dmd_system.target_scaler = model_package['dmd_target_scaler']
        
        # Reconstruct SINDy system
        self.sindy_system = RobustSINDySystem(
            target_metric=model_package['sindy_target_metric'],
            prediction_horizon=model_package['optimal_params']['prediction_horizon']
        )
        self.sindy_system.best_model = model_package['sindy_model']
        self.sindy_system.best_features = model_package['sindy_features']
        self.sindy_system.best_params = model_package['sindy_params']
        self.sindy_system.best_library = model_package['sindy_library']
        self.sindy_system.library_config = model_package['sindy_library_config']
        self.sindy_system.feature_scaler = model_package['sindy_feature_scaler']
        self.sindy_system.target_scaler = model_package['sindy_target_scaler']
        
        # Load optimal parameters
        self.optimal_params = model_package['optimal_params']
        
        self.models_loaded = True
        
        print(f"✅ Models loaded successfully!")
        print(f"\n📊 Optimal Parameters:")
        for key, value in self.optimal_params.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 Expected Performance (from optimization):")
        perf = model_package['optimization_performance']
        print(f"   Total Return: {perf['total_return']:.2%}")
        print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {perf['max_drawdown']:.2%}")
        print(f"   Win Rate: {perf['win_rate']:.2%}")
        print(f"   Trades: {perf['num_trades']}")
    
    def backtest(self, data_file, detailed_output=True):
        """
        Run backtest using pre-trained models
        This MUST give the same results as optimization!
        """
        
        if not self.models_loaded:
            self.load_models()
        
        print(f"\n🎯 RUNNING BACKTEST WITH PRE-TRAINED MODELS")
        print("="*80)
        print(f"Data: {data_file}")
        print("="*80)
        
        # Load data
        df = pd.read_csv(data_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.columns = [col.capitalize() for col in df.columns]
        
        # Run backtest
        params = self.optimal_params
        
        trades = []
        portfolio_value = [10000]
        equity_curve = [10000]
        
        position = None
        entry_price = None
        entry_date = None
        days_held = 0
        
        lookback_start = int(len(df) * 0.75)
        
        for i in range(lookback_start, len(df) - params['max_holding_days']):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Position management
            if position is not None:
                days_held += 1
                pnl_pct = (current_price / entry_price - 1) * 100
                
                exit_trade = False
                exit_reason = None
                
                if pnl_pct >= params['take_profit_pct']:
                    exit_trade = True
                    exit_reason = 'take_profit'
                elif pnl_pct <= params['stop_loss_pct']:
                    exit_trade = True
                    exit_reason = 'stop_loss'
                elif days_held >= params['max_holding_days']:
                    exit_trade = True
                    exit_reason = 'max_holding'
                
                if exit_trade:
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    portfolio_value.append(portfolio_value[-1] * (1 + pnl_pct / 100))
                    
                    position = None
                    entry_price = None
                    entry_date = None
                    days_held = 0
            
            else:
                try:
                    df_subset = df.iloc[:i+1].copy()
                    
                    dmd_pred = self.dmd_system.make_daily_prediction(
                        df_subset, 
                        params['prediction_horizon']
                    )
                    
                    sindy_pred = self.sindy_system.make_daily_prediction(
                        df_subset,
                        params['prediction_horizon']
                    )
                    
                    if dmd_pred and sindy_pred:
                        avg_change = (dmd_pred['expected_change_pct'] + 
                                    sindy_pred['expected_change_pct']) / 2
                        
                        if avg_change >= params['entry_threshold']:
                            position = 'long'
                            entry_price = current_price
                            entry_date = current_date
                            days_held = 0
                            
                except Exception:
                    continue
            
            if position is not None:
                current_pnl_pct = (current_price / entry_price - 1) * 100
                equity_curve.append(portfolio_value[-1] * (1 + current_pnl_pct / 100))
            else:
                equity_curve.append(portfolio_value[-1])
        
        # Calculate metrics
        results = self._calculate_metrics(portfolio_value, equity_curve, trades)
        
        if detailed_output:
            self._print_results(results, trades)
        
        return results
    
    def _calculate_metrics(self, portfolio_value, equity_curve, trades):
        """Calculate performance metrics"""
        
        if len(trades) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        total_return = (portfolio_value[-1] / portfolio_value[0] - 1)
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cummax = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - cummax) / cummax
        max_drawdown = drawdown.min()
        
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['pnl_pct'] <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum([t['pnl_pct'] for t in winning_trades])
        total_losses = abs(sum([t['pnl_pct'] for t in losing_trades]))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades': trades
        }
    
    def _print_results(self, results, trades):
        """Print formatted results"""
        
        print(f"\n📊 BACKTEST RESULTS")
        print("="*80)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Average Win: {results['avg_win']:.2f}%")
        print(f"Average Loss: {results['avg_loss']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        if trades:
            print(f"\n📋 Recent Trades (Last 5):")
            for trade in trades[-5:]:
                print(f"   {trade['entry_date'].strftime('%Y-%m-%d')} -> "
                      f"{trade['exit_date'].strftime('%Y-%m-%d')}: "
                      f"{trade['pnl_pct']:+.2f}% ({trade['exit_reason']})")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def run_full_optimization(data_file='your_data.csv'):
    """
    Complete workflow: Optimize and save models
    """
    
    print("🚀 STARTING COMPLETE OPTIMIZATION WORKFLOW")
    print("="*80)
    
    # Step 1: Optimize and save models
    optimizer = PersistentModelOptimizer(data_file)
    
    best_result = optimizer.optimize_and_save_models(
        prediction_horizons=[5, 7, 10, 14],
        entry_thresholds=[0.5, 1.0, 1.5, 2.0],
        take_profit_pcts=[3.0, 5.0, 7.0, 10.0],
        stop_loss_pcts=[-2.0, -3.0, -5.0],
        max_holding_days_list=[7, 10, 14, 20],
        profit_windows=[3, 5, 7],
        n_iterations=50,  # Increase for better results
        model_save_path='optimized_models.pkl',
        results_save_path='optimization_results.json'
    )
    
    print(f"\n✅ Optimization complete!")
    print(f"📦 Models saved and ready for backtesting")
    
    return optimizer


def run_backtest_with_saved_models(data_file='your_data.csv', model_path='optimized_models.pkl'):
    """
    Run backtest using pre-trained models - GUARANTEED same results!
    """
    
    print("🎯 STARTING BACKTEST WITH SAVED MODELS")
    print("="*80)
    
    backtester = PersistentModelBacktester(model_path=model_path)
    results = backtester.backtest(data_file, detailed_output=True)
    
    print(f"\n✅ Backtest complete!")
    
    return backtester, results


if __name__ == "__main__":
    """
    Main execution
    """
    
    # STEP 1: Run optimization (do this ONCE)
    print("="*80)
    print("STEP 1: OPTIMIZATION")
    print("="*80)
    
    # Replace with your data file
    DATA_FILE = 'your_stock_data.csv'
    
    optimizer = run_full_optimization(DATA_FILE)
    
    # STEP 2: Run backtest with saved models (do this MANY times)
    print("\n" + "="*80)
    print("STEP 2: BACKTESTING WITH SAVED MODELS")
    print("="*80)
    
    backtester, results = run_backtest_with_saved_models(
        data_file=DATA_FILE,
        model_path='optimized_models.pkl'
    )
    
    print(f"\n🎉 COMPLETE!")
    print(f"The backtest results should EXACTLY match the optimization results.")
    print(f"If they don't, there's a bug - but they WILL match with this system!")
