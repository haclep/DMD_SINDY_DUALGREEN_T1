"""
DAILY SIGNAL GENERATOR - PRODUCTION TRADING
============================================
Run this EVERY DAY to get fresh trading signals using your optimized models

Usage:
    python daily_signal_generator.py

This script:
1. Loads your pre-trained models (from optimization)
2. Loads latest data for your watchlist
3. Generates predictions for each stock
4. Outputs clear BUY/HOLD/SELL signals
5. Saves results to CSV with timestamp
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_PATH)

from optimize_with_model_persistence import PersistentModelBacktester


class DailySignalGenerator:
    """
    Production-ready daily signal generator
    """
    
    def __init__(self, model_path='optimized_models.pkl', data_folder='daily_data'):
        """
        Initialize the signal generator
        
        Args:
            model_path: Path to your optimized_models.pkl file
            data_folder: Folder containing daily CSV files for each stock
        """
        self.model_path = model_path
        self.data_folder = data_folder
        self.backtester = None
        self.optimal_params = None
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load the pre-trained models"""
        print("📦 Loading optimized models...")
        
        self.backtester = PersistentModelBacktester(model_path=self.model_path)
        self.backtester.load_models()
        
        self.optimal_params = self.backtester.optimal_params
        
        print("✅ Models loaded successfully!\n")
        
    def get_signal_for_stock(self, ticker, csv_file):
        """
        Get trading signal for a single stock
        
        Returns:
            dict with signal information
        """
        try:
            # Load data
            df = pd.read_csv(csv_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.columns = [col.capitalize() for col in df.columns]
            
            # Get current info
            current_date = df.index[-1]
            current_price = df['Close'].iloc[-1]
            
            # Make predictions using saved models
            dmd_pred = self.backtester.dmd_system.make_daily_prediction(
                df,
                self.optimal_params['prediction_horizon']
            )
            
            sindy_pred = self.backtester.sindy_system.make_daily_prediction(
                df,
                self.optimal_params['prediction_horizon']
            )
            
            if not dmd_pred or not sindy_pred:
                return {
                    'ticker': ticker,
                    'status': 'FAILED',
                    'signal': 'HOLD',
                    'reason': 'Prediction failed'
                }
            
            # Extract predictions
            dmd_change = dmd_pred['expected_change_pct']
            sindy_change = sindy_pred['expected_change_pct']
            avg_change = (dmd_change + sindy_change) / 2
            
            # Extract directions
            dmd_direction = dmd_pred['direction']
            sindy_direction = sindy_pred['direction']
            
            # Check agreement
            dmd_up = 'UP' in dmd_direction
            sindy_up = 'UP' in sindy_direction
            models_agree = dmd_up == sindy_up
            
            # Generate signal based on optimal parameters
            entry_threshold = self.optimal_params['entry_threshold']
            
            signal = 'HOLD'
            reason = ''
            confidence = 'LOW'
            
            if models_agree:
                if avg_change >= entry_threshold:
                    signal = 'BUY'
                    reason = f'Both models predict +{avg_change:.1f}% (threshold: {entry_threshold}%)'
                    
                    # Confidence based on how much above threshold
                    if avg_change >= entry_threshold * 2:
                        confidence = 'HIGH'
                    elif avg_change >= entry_threshold * 1.5:
                        confidence = 'MEDIUM'
                    else:
                        confidence = 'LOW'
                        
                elif avg_change <= -entry_threshold:
                    signal = 'SELL'
                    reason = f'Both models predict {avg_change:.1f}% (threshold: -{entry_threshold}%)'
                    confidence = 'MEDIUM'
                else:
                    signal = 'HOLD'
                    reason = f'Predicted move ({avg_change:+.1f}%) below threshold (±{entry_threshold}%)'
                    confidence = 'N/A'
            else:
                signal = 'HOLD'
                reason = f'Models disagree: DMD {dmd_change:+.1f}% vs SINDy {sindy_change:+.1f}%'
                confidence = 'N/A'
            
            return {
                'ticker': ticker,
                'status': 'SUCCESS',
                'date': current_date.strftime('%Y-%m-%d'),
                'price': current_price,
                'signal': signal,
                'confidence': confidence,
                'avg_predicted_change': avg_change,
                'dmd_prediction': dmd_change,
                'sindy_prediction': sindy_change,
                'models_agree': models_agree,
                'reason': reason,
                'horizon_days': self.optimal_params['prediction_horizon'],
                'take_profit': self.optimal_params['take_profit_pct'],
                'stop_loss': self.optimal_params['stop_loss_pct'],
                'max_holding_days': self.optimal_params['max_holding_days']
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'status': 'ERROR',
                'signal': 'HOLD',
                'reason': f'Error: {str(e)[:100]}'
            }
    
    def generate_daily_signals(self, watchlist):
        """
        Generate signals for all stocks in watchlist
        
        Args:
            watchlist: dict of {ticker: csv_file_path}
            
        Returns:
            DataFrame with signals for all stocks
        """
        print("🎯 GENERATING DAILY SIGNALS")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stocks to analyze: {len(watchlist)}")
        print(f"Using model: {self.model_path}")
        print("="*80)
        
        signals = []
        
        for ticker, csv_file in watchlist.items():
            print(f"\n📊 Analyzing {ticker}...")
            
            if not os.path.exists(csv_file):
                print(f"   ⚠️  File not found: {csv_file}")
                signals.append({
                    'ticker': ticker,
                    'status': 'ERROR',
                    'signal': 'HOLD',
                    'reason': 'Data file not found'
                })
                continue
            
            signal = self.get_signal_for_stock(ticker, csv_file)
            signals.append(signal)
            
            # Print result
            if signal['status'] == 'SUCCESS':
                icon = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
                print(f"   {icon} {signal['signal']}: {signal['reason']}")
                print(f"      Price: ${signal['price']:.2f} | Predicted: {signal['avg_predicted_change']:+.1f}%")
                if signal['signal'] == 'BUY':
                    print(f"      Take Profit: +{signal['take_profit']:.1f}% | Stop Loss: {signal['stop_loss']:.1f}%")
            else:
                print(f"   ❌ {signal['status']}: {signal['reason']}")
        
        # Create DataFrame
        df_signals = pd.DataFrame(signals)
        
        # Display summary
        self._display_summary(df_signals)
        
        return df_signals
    
    def _display_summary(self, df_signals):
        """Display summary of signals"""
        
        print("\n" + "="*80)
        print("📋 SIGNAL SUMMARY")
        print("="*80)
        
        # Count signals
        buy_signals = df_signals[df_signals['signal'] == 'BUY']
        sell_signals = df_signals[df_signals['signal'] == 'SELL']
        hold_signals = df_signals[df_signals['signal'] == 'HOLD']
        
        print(f"\n🟢 BUY Signals:  {len(buy_signals)}")
        print(f"🔴 SELL Signals: {len(sell_signals)}")
        print(f"⚪ HOLD Signals: {len(hold_signals)}")
        
        # Show BUY signals
        if len(buy_signals) > 0:
            print(f"\n🟢 STOCKS TO BUY:")
            print("-"*80)
            for _, row in buy_signals.iterrows():
                if row['status'] == 'SUCCESS':
                    print(f"  {row['ticker']:6s} | ${row['price']:7.2f} | "
                          f"Pred: {row['avg_predicted_change']:+5.1f}% | "
                          f"Conf: {row['confidence']:6s} | "
                          f"TP: +{row['take_profit']:.1f}% | SL: {row['stop_loss']:.1f}%")
        
        # Show SELL signals
        if len(sell_signals) > 0:
            print(f"\n🔴 STOCKS TO SELL:")
            print("-"*80)
            for _, row in sell_signals.iterrows():
                if row['status'] == 'SUCCESS':
                    print(f"  {row['ticker']:6s} | ${row['price']:7.2f} | "
                          f"Pred: {row['avg_predicted_change']:+5.1f}%")
        
        print("\n" + "="*80)
    
    def save_signals(self, df_signals, output_folder='signals'):
        """Save signals to CSV with timestamp"""
        
        # Create output folder if needed
        Path(output_folder).mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_folder}/signals_{timestamp}.csv"
        
        # Save
        df_signals.to_csv(filename, index=False)
        
        print(f"\n💾 Signals saved to: {filename}")
        
        return filename


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_daily_signals(watchlist, model_path='optimized_models.pkl'):
    """
    Main function to run daily signal generation
    
    Args:
        watchlist: dict of {ticker: csv_file_path}
        model_path: path to optimized models
    """
    
    print("\n" + "="*80)
    print("🚀 DAILY SIGNAL GENERATOR")
    print("="*80)
    
    # Initialize generator
    generator = DailySignalGenerator(model_path=model_path)
    
    # Generate signals
    signals = generator.generate_daily_signals(watchlist)
    
    # Save to CSV
    output_file = generator.save_signals(signals)
    
    # Also save latest signals (overwrite)
    signals.to_csv('signals/latest_signals.csv', index=False)
    print(f"💾 Latest signals saved to: signals/latest_signals.csv")
    
    print("\n✅ Daily signal generation complete!")
    print("="*80)
    
    return signals


# =============================================================================
# CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    """
    Configure your watchlist here and run daily
    """
    
    # CONFIGURATION: YOUR WATCHLIST
    # Format: {ticker: path_to_csv_file}
    
    WATCHLIST = {
        'IONQ':  '110725/IONQ_daily.csv',
        'PLTR':  'daily_data/PLTR_daily.csv',
        'AMD':   'daily_data/AMD_daily.csv',
        'ENPH':  'daily_data/ENPH_daily.csv',
        'COIN':  'daily_data/COIN_daily.csv',
        'DKNG':  'daily_data/DKNG_daily.csv',
        # Add more stocks here...
    }
    
    # Path to your optimized models
    MODEL_PATH = 'optimized_models.pkl'
    
    # Run signal generation
    signals = run_daily_signals(WATCHLIST, MODEL_PATH)
    
    # Optional: Print to console in clean format
    print("\n" + "="*80)
    print("📊 TODAY'S TRADING SIGNALS")
    print("="*80)
    
    buy_signals = signals[signals['signal'] == 'BUY']
    if len(buy_signals) > 0:
        print("\n🟢 BUY THESE STOCKS:")
        for _, row in buy_signals.iterrows():
            if row['status'] == 'SUCCESS':
                print(f"  • {row['ticker']:6s} @ ${row['price']:7.2f} | "
                      f"Expected: {row['avg_predicted_change']:+.1f}% | "
                      f"TP: +{row['take_profit']:.1f}% | SL: {row['stop_loss']:.1f}%")
    else:
        print("\n⚪ No BUY signals today")
    
    print("\n" + "="*80)
