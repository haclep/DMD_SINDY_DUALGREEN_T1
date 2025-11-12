# Individual DMD + SINDy Trading Analyzer (No Weighted Combination)
# Each method provides independent analysis and recommendations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Simplified output classes for clean results
class TradingInsight:
    """Clean trading insight with essential information only"""
    
    def __init__(self, ticker: str, file_path: str):
        self.ticker = ticker
        self.file_path = file_path
        self.current_date = None
        self.current_price = None
        
        # Individual model predictions (no combination)
        self.dmd_prediction = None
        self.sindy_prediction = None
        
        # Individual accuracy metrics
        self.dmd_accuracy = None
        self.sindy_accuracy = None
        
        # Individual recommendations (separate)
        self.dmd_recommendation = None
        self.sindy_recommendation = None
        
    def to_dict(self):
        """Convert to dictionary for easy export"""
        result = {
            'ticker': self.ticker,
            'current_date': self.current_date.strftime('%Y-%m-%d') if self.current_date else None,
            'current_price': round(self.current_price, 2) if self.current_price else None,
            'dmd_accuracy': f"{self.dmd_accuracy:.1%}" if self.dmd_accuracy else None,
            'sindy_accuracy': f"{self.sindy_accuracy:.1%}" if self.sindy_accuracy else None
        }
        
        # Add DMD results
        if self.dmd_prediction:
            result.update({
                'dmd_direction': self.dmd_prediction.get('direction', 'N/A'),
                'dmd_signal': self.dmd_prediction.get('signal', 'N/A'),
                'dmd_confidence': self.dmd_prediction.get('confidence', 'N/A'),
                'dmd_5d_change': self.dmd_prediction.get('projected_change', 'N/A'),
                'dmd_action': self.dmd_prediction.get('action', 'N/A')
            })
        
        # Add SINDy results
        if self.sindy_prediction:
            result.update({
                'sindy_direction': self.sindy_prediction.get('direction', 'N/A'),
                'sindy_signal': self.sindy_prediction.get('signal', 'N/A'),
                'sindy_confidence': self.sindy_prediction.get('confidence', 'N/A'),
                'sindy_5d_change': self.sindy_prediction.get('projected_change', 'N/A'),
                'sindy_action': self.sindy_prediction.get('action', 'N/A')
            })
        
        return result

class SimplifiedDMDRunner:
    """Simplified DMD runner that outputs only essential information"""
    
    def __init__(self, target_metric='log_price_ratio_sma20', prediction_horizon=5):
        self.target_metric = target_metric
        self.prediction_horizon = prediction_horizon  # NEW
        
    def run_analysis(self, df: pd.DataFrame, ticker: str) -> Tuple[Optional[Dict], Optional[float]]:
        """Run DMD analysis and return simplified results"""
        try:
            print(f"   🔵 Running DMD (Linear) analysis for {self.prediction_horizon}-day horizon...")  # UPDATED
            
            # Initialize and run DMD system
            dmd_system = RobustDMDSystem(target_metric=self.target_metric, prediction_horizon=self.prediction_horizon)  # CHANGED
            
            # Quick optimization (reduced trials for speed)
            dmd_system.comprehensive_optimize(df, max_features=6, n_feature_trials=20)
            
            # Training and evaluation
            predictions, actuals, directional_results = dmd_system.final_training_and_evaluation(df)
            
            # Get best accuracy
            best_horizon = max(directional_results.keys(), 
                              key=lambda h: directional_results[h]['accuracy_non_neutral'] 
                              if not np.isnan(directional_results[h]['accuracy_non_neutral']) 
                              else directional_results[h]['accuracy_all'])
            
            best_acc = directional_results[best_horizon]['accuracy_non_neutral']
            if np.isnan(best_acc):
                best_acc = directional_results[best_horizon]['accuracy_all']
            
            # Make prediction
            live_prediction = dmd_system.make_daily_prediction(df, self.prediction_horizon)  # CHANGED
            
            # Parse direction and signal
            direction_raw = live_prediction['direction']
            direction = direction_raw.replace('📈 ', '').replace('📉 ', '').replace('➡️ ', '')
            
            if 'UP' in direction:
                signal = "BULLISH"
            elif 'DOWN' in direction:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            # Calculate projected 5-day change
            expected_change = live_prediction['expected_change_pct']
            projected_5d = expected_change  # Direct 5-day prediction
            
            # Generate action recommendation
            if abs(expected_change) > 1 and live_prediction['confidence'] in ['HIGH', 'MEDIUM']:
                action = "BUY" if expected_change > 0 else "SELL"
            else:
                action = "HOLD"
            
            # Simplified output
            result = {
                'direction': direction,
                'signal': signal,
                'confidence': live_prediction['confidence'],
                'expected_change_pct': expected_change,
                'projected_change': f"{projected_5d:+.1f}%",
                'action': action,
                'target_metric': self.target_metric,
                'best_horizon': best_horizon,
                'model_type': 'LINEAR_DMD'
            }
            
            return result, best_acc
            
        except Exception as e:
            print(f"     ❌ DMD failed: {str(e)[:100]}...")
            return None, None

class SimplifiedSINDyRunner:
    """Simplified SINDy runner that outputs only essential information"""
    
    def __init__(self, target_metric='log_price_ratio_sma20', prediction_horizon=5):
        self.target_metric = target_metric
        self.prediction_horizon = prediction_horizon  # NEW

    def run_analysis(self, df: pd.DataFrame, ticker: str) -> Tuple[Optional[Dict], Optional[float]]:
        """Run SINDy analysis and return simplified results"""
        try:
            print(f"   🔴 Running SINDy (Nonlinear) analysis for {self.prediction_horizon}-day horizon...")  # UPDATED
            
            # Initialize and run SINDy system
            sindy_system = RobustSINDySystem(target_metric=self.target_metric, prediction_horizon=self.prediction_horizon)  # CHANGED
            
            # Quick optimization (reduced trials for speed)
            sindy_system.comprehensive_optimize(df, max_features=8, n_feature_trials=50)
            
            # Training and evaluation
            predictions, actuals, directional_results = sindy_system.final_training_and_evaluation(df)
            
            # Get best accuracy
            best_horizon = max(directional_results.keys(), 
                              key=lambda h: directional_results[h]['accuracy_non_neutral'] 
                              if not np.isnan(directional_results[h]['accuracy_non_neutral']) 
                              else directional_results[h]['accuracy_all'])
            
            best_acc = directional_results[best_horizon]['accuracy_non_neutral']
            if np.isnan(best_acc):
                best_acc = directional_results[best_horizon]['accuracy_all']
            
            # Make prediction
            live_prediction = sindy_system.make_daily_prediction(df, self.prediction_horizon)  # CHANGED
            
            # Parse direction and signal
            direction_raw = live_prediction['direction']
            direction = direction_raw.replace('📈 ', '').replace('📉 ', '').replace('➡️ ', '')
            
            if 'UP' in direction:
                signal = "BULLISH"
            elif 'DOWN' in direction:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            # Calculate projected 5-day change
            expected_change = live_prediction['expected_change_pct']
            projected_5d = expected_change # Direct 5-day prediction
            
            # Generate action recommendation
            if abs(expected_change) > 1 and live_prediction['confidence'] in ['HIGH', 'MEDIUM']:
                action = "BUY" if expected_change > 0 else "SELL"
            else:
                action = "HOLD"
            
            # Get key nonlinear factors
            important_terms = sindy_system.get_important_terms(top_n=3)
            nonlinear_factors = [term['feature'] for term in important_terms 
                               if any(x in term['feature'] for x in ['^2', '*', '/', 'sqrt'])]
            
            # Simplified output
            result = {
                'direction': direction,
                'signal': signal,
                'confidence': live_prediction['confidence'],
                'expected_change_pct': expected_change,
                'projected_change': f"{projected_5d:+.1f}%",
                'action': action,
                'target_metric': self.target_metric,
                'best_horizon': best_horizon,
                'nonlinear_factors': nonlinear_factors[:2],  # Top 2 nonlinear terms
                'model_type': 'NONLINEAR_SINDY'
            }
            
            return result, best_acc
            
        except Exception as e:
            print(f"     ❌ SINDy failed: {str(e)[:100]}...")
            return None, None

class IndividualMethodsAnalyzer:
    """Main analyzer that runs DMD and SINDy as separate individual methods"""
    
    def __init__(self, target_metric='log_price_ratio_sma20', prediction_horizon=5):
        self.target_metric = target_metric
        self.prediction_horizon = prediction_horizon  # NEW
        self.dmd_runner = SimplifiedDMDRunner(target_metric, prediction_horizon) if DMD_AVAILABLE else None  # CHANGED
        self.sindy_runner = SimplifiedSINDyRunner(target_metric, prediction_horizon) if SINDY_AVAILABLE else None  # CHANGED
    
    def load_and_validate_data(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # Standardize column names
            df.columns = [col.capitalize() for col in df.columns]
            
            # Ensure we have required columns
            required_cols = ['Close', 'High', 'Low', 'Open']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Try to infer from available columns
                if 'Close' not in df.columns:
                    return None, f"Missing Close price column"
                
                # Fill missing OHLC with Close
                for col in missing_cols:
                    df[col] = df['Close']
            
            # Basic validation
            if len(df) < 100:
                return None, f"Insufficient data: only {len(df)} rows"
            
            # Extract ticker from filename
            ticker = Path(file_path).stem.upper()
            
            return df, ticker
            
        except Exception as e:
            return None, f"Failed to load data: {str(e)}"
    
    def generate_trading_insight(self, df: pd.DataFrame, ticker: str, 
                               dmd_result: Optional[Dict], dmd_accuracy: Optional[float],
                               sindy_result: Optional[Dict], sindy_accuracy: Optional[float]) -> TradingInsight:
        """Generate clean trading insight from individual model results"""
        
        insight = TradingInsight(ticker, "")
        insight.current_date = df.index[-1]
        insight.current_price = df['Close'].iloc[-1]
        insight.dmd_accuracy = dmd_accuracy
        insight.sindy_accuracy = sindy_accuracy
        
        # Store individual predictions (no combination)
        insight.dmd_prediction = dmd_result
        insight.sindy_prediction = sindy_result
        
        return insight
    
    def analyze_single_file(self, file_path: str) -> TradingInsight:
        """Analyze a single CSV file and return trading insight"""
        
        print(f"\n📈 Analyzing: {Path(file_path).name}")
        
        # Load and validate data
        df, ticker = self.load_and_validate_data(file_path)
        if df is None:
            print(f"   ❌ Skipped: {ticker}")
            return TradingInsight(ticker or "UNKNOWN", file_path)
        
        print(f"   📊 Loaded {len(df)} days of data for {ticker}")
        print(f"   📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Run DMD analysis
        dmd_result, dmd_accuracy = None, None
        if self.dmd_runner:
            dmd_result, dmd_accuracy = self.dmd_runner.run_analysis(df, ticker)
        
        # Run SINDy analysis
        sindy_result, sindy_accuracy = None, None
        if self.sindy_runner:
            sindy_result, sindy_accuracy = self.sindy_runner.run_analysis(df, ticker)
        
        # Generate trading insight
        insight = self.generate_trading_insight(df, ticker, dmd_result, dmd_accuracy, sindy_result, sindy_accuracy)
        
        # Show individual results
        if dmd_result:
            print(f"   🔵 DMD: {dmd_result['signal']} {dmd_result['direction']} ({dmd_result['confidence']}) - {dmd_result['action']}")
        if sindy_result:
            print(f"   🔴 SINDy: {sindy_result['signal']} {sindy_result['direction']} ({sindy_result['confidence']}) - {sindy_result['action']}")
        
        print(f"   ✅ Analysis complete")
        
        return insight
    
    def analyze_folder(self, folder_path: str) -> List[TradingInsight]:
        """Analyze all CSV files in a folder"""
        
        folder = Path(folder_path)
        csv_files = list(folder.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {folder_path}")
            return []
        
        print(f"🔍 Found {len(csv_files)} CSV files in {folder_path}")
        
        insights = []
        for file_path in csv_files:
            try:
                insight = self.analyze_single_file(str(file_path))
                insights.append(insight)
            except Exception as e:
                print(f"   ❌ Error analyzing {file_path.name}: {str(e)[:100]}...")
                # Create failed insight
                failed_insight = TradingInsight(file_path.stem.upper(), str(file_path))
                insights.append(failed_insight)
        
        return insights
    
    def analyze_multiple_folders(self, folder_paths: List[str]) -> Dict[str, List[TradingInsight]]:
        """Analyze CSV files in multiple folders"""
        
        all_insights = {}
        
        for folder_path in folder_paths:
            folder_name = Path(folder_path).name
            print(f"\n{'='*60}")
            print(f"📁 ANALYZING FOLDER: {folder_name}")
            print(f"{'='*60}")
            
            insights = self.analyze_folder(folder_path)
            all_insights[folder_name] = insights
        
        return all_insights

def display_individual_insights_summary(insights: List[TradingInsight], folder_name: str = ""):
    """Display clean summary of individual trading insights"""
    
    if folder_name:
        print(f"\n📊 INDIVIDUAL METHODS SUMMARY - {folder_name}")
    else:
        print(f"\n📊 INDIVIDUAL METHODS SUMMARY")
    print("="*100)
    
    if not insights:
        print("❌ No insights to display")
        return
    
    # Create summary table
    summary_data = []
    for insight in insights:
        
        # DMD results
        dmd_direction = insight.dmd_prediction['direction'] if insight.dmd_prediction else "N/A"
        dmd_signal = insight.dmd_prediction['signal'] if insight.dmd_prediction else "N/A"
        dmd_confidence = insight.dmd_prediction['confidence'] if insight.dmd_prediction else "N/A"
        dmd_change = insight.dmd_prediction['projected_change'] if insight.dmd_prediction else "N/A"
        dmd_action = insight.dmd_prediction['action'] if insight.dmd_prediction else "N/A"
        
        # SINDy results
        sindy_direction = insight.sindy_prediction['direction'] if insight.sindy_prediction else "N/A"
        sindy_signal = insight.sindy_prediction['signal'] if insight.sindy_prediction else "N/A"
        sindy_confidence = insight.sindy_prediction['confidence'] if insight.sindy_prediction else "N/A"
        sindy_change = insight.sindy_prediction['projected_change'] if insight.sindy_prediction else "N/A"
        sindy_action = insight.sindy_prediction['action'] if insight.sindy_prediction else "N/A"
        
        summary_data.append({
            'Ticker': insight.ticker,
            'Date': insight.current_date.strftime('%Y-%m-%d') if insight.current_date else "N/A",
            'Price': f"${insight.current_price:.2f}" if insight.current_price else "N/A",
            
            # DMD columns
            'DMD_Dir': dmd_direction,
            'DMD_Signal': dmd_signal,
            'DMD_Conf': dmd_confidence,
            'DMD_5D': dmd_change,
            'DMD_Action': dmd_action,
            'DMD_Acc': f"{insight.dmd_accuracy:.1%}" if insight.dmd_accuracy else "N/A",
            
            # SINDy columns  
            'SINDy_Dir': sindy_direction,
            'SINDy_Signal': sindy_signal,
            'SINDy_Conf': sindy_confidence,
            'SINDy_5D': sindy_change,
            'SINDy_Action': sindy_action,
            'SINDy_Acc': f"{insight.sindy_accuracy:.1%}" if insight.sindy_accuracy else "N/A"
        })
    
    # Convert to DataFrame for nice display
    df_summary = pd.DataFrame(summary_data)
    
    # Display with formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 12)
    
    print(df_summary.to_string(index=False))
    
    # Quick stats by method
    print(f"\n📈 DMD (LINEAR) STATS:")
    dmd_valid = [i for i in insights if i.dmd_prediction]
    if dmd_valid:
        dmd_bullish = sum(1 for i in dmd_valid if i.dmd_prediction['signal'] == "BULLISH")
        dmd_bearish = sum(1 for i in dmd_valid if i.dmd_prediction['signal'] == "BEARISH")
        dmd_neutral = sum(1 for i in dmd_valid if i.dmd_prediction['signal'] == "NEUTRAL")
        dmd_high_conf = sum(1 for i in dmd_valid if i.dmd_prediction['confidence'] == "HIGH")
        dmd_avg_acc = np.mean([i.dmd_accuracy for i in dmd_valid if i.dmd_accuracy])
        
        print(f"   Analyzed: {len(dmd_valid)} tickers")
        print(f"   Bullish: {dmd_bullish} | Bearish: {dmd_bearish} | Neutral: {dmd_neutral}")
        print(f"   High Confidence: {dmd_high_conf} ({dmd_high_conf/len(dmd_valid):.1%})")
        print(f"   Average Accuracy: {dmd_avg_acc:.1%}")
    else:
        print("   No valid DMD results")
    
    print(f"\n🔬 SINDY (NONLINEAR) STATS:")
    sindy_valid = [i for i in insights if i.sindy_prediction]
    if sindy_valid:
        sindy_bullish = sum(1 for i in sindy_valid if i.sindy_prediction['signal'] == "BULLISH")
        sindy_bearish = sum(1 for i in sindy_valid if i.sindy_prediction['signal'] == "BEARISH")
        sindy_neutral = sum(1 for i in sindy_valid if i.sindy_prediction['signal'] == "NEUTRAL")
        sindy_high_conf = sum(1 for i in sindy_valid if i.sindy_prediction['confidence'] == "HIGH")
        sindy_avg_acc = np.mean([i.sindy_accuracy for i in sindy_valid if i.sindy_accuracy])
        
        print(f"   Analyzed: {len(sindy_valid)} tickers")
        print(f"   Bullish: {sindy_bullish} | Bearish: {sindy_bearish} | Neutral: {sindy_neutral}")
        print(f"   High Confidence: {sindy_high_conf} ({sindy_high_conf/len(sindy_valid):.1%})")
        print(f"   Average Accuracy: {sindy_avg_acc:.1%}")
    else:
        print("   No valid SINDy results")

def display_detailed_individual_insight(insight: TradingInsight):
    """Display detailed information for a single ticker with individual methods"""
    
    print(f"\n{'='*60}")
    print(f"🎯 DETAILED INDIVIDUAL ANALYSIS: {insight.ticker}")
    print(f"{'='*60}")
    
    print(f"📅 Current Status ({insight.current_date.strftime('%Y-%m-%d') if insight.current_date else 'N/A'}):")
    print(f"   Current Price: ${insight.current_price:.2f}" if insight.current_price else "   Current Price: N/A")
    
    # DMD Analysis
    if insight.dmd_prediction:
        print(f"\n🔵 DMD (LINEAR) ANALYSIS:")
        print(f"   Direction: {insight.dmd_prediction['direction']}")
        print(f"   Signal: {insight.dmd_prediction['signal']}")
        print(f"   Confidence: {insight.dmd_prediction['confidence']}")
        print(f"   5-Day Projection: {insight.dmd_prediction['projected_change']}")
        print(f"   Recommended Action: {insight.dmd_prediction['action']}")
        print(f"   Model Accuracy: {insight.dmd_accuracy:.1%}" if insight.dmd_accuracy else "   Model Accuracy: N/A")
    else:
        print(f"\n🔵 DMD (LINEAR) ANALYSIS: ❌ Failed")
    
    # SINDy Analysis
    if insight.sindy_prediction:
        print(f"\n🔴 SINDY (NONLINEAR) ANALYSIS:")
        print(f"   Direction: {insight.sindy_prediction['direction']}")
        print(f"   Signal: {insight.sindy_prediction['signal']}")
        print(f"   Confidence: {insight.sindy_prediction['confidence']}")
        print(f"   5-Day Projection: {insight.sindy_prediction['projected_change']}")
        print(f"   Recommended Action: {insight.sindy_prediction['action']}")
        print(f"   Model Accuracy: {insight.sindy_accuracy:.1%}" if insight.sindy_accuracy else "   Model Accuracy: N/A")
        
        # Show nonlinear factors
        if 'nonlinear_factors' in insight.sindy_prediction and insight.sindy_prediction['nonlinear_factors']:
            print(f"   Key Nonlinear Terms: {', '.join(insight.sindy_prediction['nonlinear_factors'])}")
    else:
        print(f"\n🔴 SINDY (NONLINEAR) ANALYSIS: ❌ Failed")
    
    # Agreement/Disagreement Analysis
    if insight.dmd_prediction and insight.sindy_prediction:
        dmd_signal = insight.dmd_prediction['signal']
        sindy_signal = insight.sindy_prediction['signal']
        
        print(f"\n⚖️  MODEL COMPARISON:")
        if dmd_signal == sindy_signal:
            print(f"   ✅ AGREEMENT: Both models predict {dmd_signal}")
            print(f"   💡 Strong signal when linear and nonlinear agree")
        else:
            print(f"   ⚠️  DISAGREEMENT: DMD={dmd_signal}, SINDy={sindy_signal}")
            print(f"   💡 Consider market regime transition")

def export_individual_insights_to_csv(insights: List[TradingInsight], filename: str):
    """Export individual insights to CSV file"""
    
    data = [insight.to_dict() for insight in insights]
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)
    print(f"✅ Individual insights exported to: {filename}")

# Utility functions for individual methods
def quick_individual_analysis(file_path: str, target_metric: str = 'log_price_ratio_sma20', 
                             prediction_horizon: int = 5) -> TradingInsight:  # ADD prediction_horizon
    """Quick analysis of a single file with individual methods"""
    analyzer = IndividualMethodsAnalyzer(target_metric=target_metric, prediction_horizon=prediction_horizon)  # CHANGED
    return analyzer.analyze_single_file(file_path)

def batch_individual_analysis(folder_path: str, target_metric: str = 'log_price_ratio_sma20', 
                             prediction_horizon: int = 5) -> List[TradingInsight]:  # ADD prediction_horizon
    """Batch analysis of a folder with individual methods"""
    analyzer = IndividualMethodsAnalyzer(target_metric=target_metric, prediction_horizon=prediction_horizon)  # CHANGED
    insights = analyzer.analyze_folder(folder_path)
    display_individual_insights_summary(insights, Path(folder_path).name)
    return insights

# Example usage and main execution
if __name__ == "__main__":
    
    print("🚀 INDIVIDUAL METHODS DMD + SINDY ANALYZER")
    print("="*80)
    
    # Check system availability
    if DMD_AVAILABLE and SINDY_AVAILABLE:
        print("✅ Both DMD and SINDy systems available")
    elif SINDY_AVAILABLE:
        print("⚠️  Only SINDy system available")
    elif DMD_AVAILABLE:
        print("⚠️  Only DMD system available")
    else:
        print("❌ No analysis systems available")
        exit(1)
    
    # Configuration
    TARGET_METRIC = 'log_price_ratio_sma20'  # Change as needed
    PREDICTION_HORIZON = 5  # NEW: Add this configuration

    
    # Initialize analyzer (no weights needed!)
    analyzer = IndividualMethodsAnalyzer(target_metric=TARGET_METRIC, prediction_horizon=PREDICTION_HORIZON)  # CHANGED
    
    print(f"🎯 Target Metric: {TARGET_METRIC}")
    print(f"📅 Prediction Horizon: {PREDICTION_HORIZON} days")  # NEW
    print(f"📊 Individual Methods: DMD (Linear) + SINDy (Nonlinear)")
    
    # CONFIGURATION: Set your folder paths here
    folder_paths = [
        "data/folder1",  # Replace with your actual folder paths
        "data/folder2",  # Add more folders as needed
        "data/folder3"
    ]
    
    # Multiple folders analysis
    all_insights = analyzer.analyze_multiple_folders(folder_paths)
    
    # Display results for each folder
    for folder_name, insights in all_insights.items():
        display_individual_insights_summary(insights, folder_name)
        
        # Export each folder's results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"individual_methods_{folder_name}_{timestamp}.csv"
        export_individual_insights_to_csv(insights, export_filename)
    
    # Display high-confidence signals from either method
    print(f"\n{'='*80}")
    print("🔍 HIGH-CONFIDENCE SIGNALS (Individual Methods)")
    print("="*80)
    
    high_confidence_insights = []
    for folder_name, insights in all_insights.items():
        for insight in insights:
            # Check DMD high confidence
            if (insight.dmd_prediction and 
                insight.dmd_prediction['confidence'] == "HIGH" and 
                insight.dmd_prediction['signal'] in ["BULLISH", "BEARISH"]):
                high_confidence_insights.append(('DMD', insight))
            
            # Check SINDy high confidence
            if (insight.sindy_prediction and 
                insight.sindy_prediction['confidence'] == "HIGH" and 
                insight.sindy_prediction['signal'] in ["BULLISH", "BEARISH"]):
                high_confidence_insights.append(('SINDy', insight))
    
    if high_confidence_insights:
        print(f"Found {len(high_confidence_insights)} high-confidence signals:")
        for method, insight in high_confidence_insights:
            if method == 'DMD':
                pred = insight.dmd_prediction
            else:
                pred = insight.sindy_prediction
            print(f"   🎯 {insight.ticker} ({method}): {pred['signal']} {pred['direction']} - {pred['action']}")
    else:
        print("No high-confidence signals found from either method.")
    
    # Summary of all folders combined
    all_combined_insights = []
    for insights in all_insights.values():
        all_combined_insights.extend(insights)
    
    if all_combined_insights:
        print(f"\n{'='*80}")
        print("📊 OVERALL INDIVIDUAL METHODS SUMMARY")
        print("="*80)
        
        # DMD summary
        dmd_results = [i for i in all_combined_insights if i.dmd_prediction]
        dmd_buy = len([i for i in dmd_results if i.dmd_prediction['action'] == 'BUY'])
        dmd_sell = len([i for i in dmd_results if i.dmd_prediction['action'] == 'SELL'])
        dmd_hold = len([i for i in dmd_results if i.dmd_prediction['action'] == 'HOLD'])
        
        # SINDy summary
        sindy_results = [i for i in all_combined_insights if i.sindy_prediction]
        sindy_buy = len([i for i in sindy_results if i.sindy_prediction['action'] == 'BUY'])
        sindy_sell = len([i for i in sindy_results if i.sindy])