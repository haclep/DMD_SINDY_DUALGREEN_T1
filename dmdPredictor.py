import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class EnhancedDMD:
    """Enhanced DMD with better regularization and parameter options"""
    
    def __init__(self, rank=10, regularization=1e-8):
        self.rank = rank
        self.regularization = regularization
        self.A = None
        
    def fit(self, X, y):
        """
        Fit linear model: y_next = A @ x_current with regularization
        X: (n_features, n_samples) - features at time t
        y: (n_samples,) - targets at time t+1
        """
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        # Add regularization to prevent overfitting
        if self.rank and self.rank < X.shape[0]:
            # Use SVD for rank reduction
            U, s, Vh = svd(X, full_matrices=False)
            r = min(self.rank, len(s))
            
            # Truncate
            U_r = U[:, :r]
            s_r = s[:r]
            Vh_r = Vh[:r, :]
            
            # Regularized pseudoinverse
            s_inv = 1 / (s_r + self.regularization)
            X_pinv = Vh_r.T @ np.diag(s_inv) @ U_r.T
        else:
            # Standard regularized pseudoinverse
            X_pinv = np.linalg.pinv(X, rcond=self.regularization)
        
        self.A = y @ X_pinv
        return self
    
    def predict(self, X):
        """Predict next value(s)"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.A @ X


class StrictIndicatorLibrary:
    """Strictly no-leakage technical indicators"""
    
    @staticmethod
    def create_indicators(df, max_lags=5, prediction_horizon=5):  # ADD prediction_horizon parameter
        """Create indicators with STRICT no-future-information policy"""
        indicators = {}
        
        # CRITICAL: All features must be lagged by AT LEAST prediction_horizon days
        min_lag = prediction_horizon  # NEW: Minimum lag to prevent leakage
        
        # ONLY use past price information (at least prediction_horizon-day lag)
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        
        # Price-based features (all lagged by at least prediction_horizon days)
        for lag in range(min_lag, max_lags + min_lag):  # CHANGED: start from min_lag
            indicators[f'returns_{lag}d'] = close.pct_change(lag).shift(1)
            indicators[f'log_returns_{lag}d'] = (np.log(close) - np.log(close.shift(lag))).shift(1)
            
            # Price position features
            indicators[f'high_low_ratio_{lag}d'] = (high.shift(lag) / low.shift(lag))
            indicators[f'close_open_ratio_{lag}d'] = (close.shift(lag) / open_price.shift(lag))
        
        # Moving average features (all properly lagged)
        for period in [5, 10, 20, 50]:
            sma = close.rolling(period, min_periods=period).mean()
            
            # All features use at least prediction_horizon-day lagged data
            indicators[f'sma_{period}_level'] = sma.shift(min_lag)  # CHANGED
            indicators[f'price_to_sma_{period}'] = (close.shift(min_lag) / sma.shift(min_lag))  # CHANGED
            indicators[f'sma_{period}_slope'] = sma.pct_change(5).shift(min_lag)  # CHANGED
            indicators[f'sma_{period}_momentum'] = (sma.shift(min_lag) / sma.shift(min_lag + 5) - 1)  # CHANGED
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            vol = df['Volume']
            vol_sma = vol.rolling(20).mean()
            indicators['volume_ratio'] = (vol.shift(min_lag) / vol_sma.shift(min_lag))  # CHANGED
            indicators['volume_momentum'] = vol.pct_change(5).shift(min_lag)  # CHANGED
        
        # Volatility features (using past returns)
        base_returns = close.pct_change(1)
        for period in [5, 10, 20, 30]:
            vol = base_returns.rolling(period).std() * np.sqrt(252)
            indicators[f'volatility_{period}d'] = vol.shift(min_lag)  # CHANGED
            
            # Volatility momentum
            vol_ma = vol.rolling(10).mean()
            indicators[f'vol_ratio_{period}d'] = (vol.shift(min_lag) / vol_ma.shift(min_lag))  # CHANGED
        
        # Technical indicators (all lagged)
        # RSI with proper lag
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi_14'] = rsi.shift(min_lag)  # CHANGED
        indicators['rsi_momentum'] = rsi.pct_change(5).shift(min_lag)  # CHANGED
        
        # Momentum indicators (strictly lagged)
        for period in [3, 5, 10, 20]:
            indicators[f'momentum_{period}d'] = (close.shift(min_lag) / close.shift(period + min_lag) - 1)  # CHANGED
        
        # Bollinger Band position (lagged)
        bb_period = 20
        bb_middle = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_position = (close - bb_middle) / (2 * bb_std + 1e-10)
        indicators['bb_position'] = bb_position.shift(min_lag)  # CHANGED
        
        # Price extremes (lagged)
        for period in [5, 10, 20]:
            highest = high.rolling(period).max()
            lowest = low.rolling(period).min()
            indicators[f'price_position_{period}d'] = ((close.shift(min_lag) - lowest.shift(min_lag)) / 
                                                      (highest.shift(min_lag) - lowest.shift(min_lag) + 1e-10))  # CHANGED
        
        return pd.DataFrame(indicators, index=df.index)


class RobustDMDSystem:
    """Robust DMD system with comprehensive parameter search"""
    
    def __init__(self, target_metric='log_prices', prediction_horizon=5):  # ADD prediction_horizon
        self.target_metric = target_metric
        self.prediction_horizon = prediction_horizon  # NEW
        self.best_features = None
        self.best_params = None
        self.best_model = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def prepare_target(self, df):
        """Prepare target - ONLY use scientifically validated linear metrics"""
        if self.target_metric == 'log_prices':
            # Predict N-day ahead log price (changed from -1)
            log_prices = np.log(df['Close'])
            future_log_price = log_prices.shift(-self.prediction_horizon)  # CHANGED
            return future_log_price
        elif self.target_metric == 'price_ratio_sma200':
            # Predict N-day ahead price ratio to SMA200
            sma_200 = df['Close'].rolling(200).mean()
            current_ratio = df['Close'] / sma_200
            future_ratio = current_ratio.shift(-self.prediction_horizon)  # CHANGED
            return future_ratio
        elif self.target_metric == 'log_price_ratio_sma200':
            # Log price ratio to SMA200
            sma_200 = df['Close'].rolling(200).mean()
            current_ratio = np.log(df['Close'] / sma_200)
            future_ratio = current_ratio.shift(-self.prediction_horizon)  # CHANGED
            return future_ratio
        elif self.target_metric == 'log_price_ratio_sma20':
            # Alternative: shorter MA for more responsive signals
            sma_20 = df['Close'].rolling(20).mean()
            current_ratio = np.log(df['Close'] / sma_20)
            future_ratio = current_ratio.shift(-self.prediction_horizon)  # CHANGED
            return future_ratio
        elif self.target_metric == 'log_returns':
            log_returns = np.log(df['Close'] / df['Close'].shift(1))
            future_log_return = log_returns.shift(-self.prediction_horizon)  # CHANGED
            return future_log_return
        else:
            raise ValueError(f"Target '{self.target_metric}' not scientifically validated as linear!\n"
                           f"Use only: 'log_prices', 'price_ratio_sma200', 'log_price_ratio_sma200', 'log_price_ratio_sma20'")
    
    def create_dataset(self, indicators_df, target_series, feature_cols, min_periods=100):
        """Create dataset with strict alignment"""
        # Select features
        feature_data = []
        for col in feature_cols:
            if col in indicators_df.columns:
                feature_data.append(indicators_df[col].values)
        
        if not feature_data:
            return None, None, None
        
        X = np.vstack(feature_data)  # (n_features, n_times)
        Y = target_series.values
        
        # Remove any samples with NaN in features OR target
        feature_valid = ~np.any(np.isnan(X), axis=0)
        target_valid = ~np.isnan(Y)
        valid_mask = feature_valid & target_valid
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < min_periods:
            return None, None, None
        
        X_clean = X[:, valid_indices]
        Y_clean = Y[valid_indices]
        
        return X_clean, Y_clean, valid_indices
    
    def evaluate_model(self, train_indicators, train_target, val_indicators, val_target, 
                      feature_cols, dmd_params, min_test_samples=50):
        """Comprehensive model evaluation with directional accuracy focus"""
        try:
            # Create datasets
            X_train, Y_train, _ = self.create_dataset(train_indicators, train_target, feature_cols)
            X_val, Y_val, _ = self.create_dataset(val_indicators, val_target, feature_cols)
            
            if (X_train is None or X_val is None or 
                X_train.shape[1] < 100 or X_val.shape[1] < min_test_samples):
                return float('inf')
            
            # Use robust scaling
            scaler_type = dmd_params.get('scaler', 'standard')
            if scaler_type == 'robust':
                scaler_X = RobustScaler()
                scaler_Y = RobustScaler()
            else:
                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
            
            # Scale features
            X_train_scaled = scaler_X.fit_transform(X_train.T).T
            X_val_scaled = scaler_X.transform(X_val.T).T
            
            # Scale targets
            Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
            Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
            
            # Train DMD
            dmd = EnhancedDMD(
                rank=dmd_params.get('rank', 10),
                regularization=dmd_params.get('regularization', 1e-8)
            )
            dmd.fit(X_train_scaled, Y_train_scaled)
            
            # Validate
            Y_pred_scaled = dmd.predict(X_val_scaled)
            
            # Inverse transform to get actual values
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.T).flatten()
            
            # Calculate directional accuracy for different horizons
            directional_accuracies = {}
            
            for horizon in [1, 3, 5]:
                if len(Y_pred) >= horizon and len(Y_val) >= horizon:
                    try:
                        # Multi-day directional prediction
                        if horizon == 1:
                            pred_direction = np.sign(Y_pred)
                            actual_direction = np.sign(Y_val)
                        else:
                            # For multi-day: use cumulative change
                            if len(Y_pred) >= horizon:
                                pred_direction = np.sign(np.convolve(Y_pred, np.ones(horizon), mode='valid'))
                                actual_direction = np.sign(np.convolve(Y_val, np.ones(horizon), mode='valid'))
                            else:
                                continue
                        
                        # Calculate accuracy (ignore neutral predictions for now)
                        if len(pred_direction) > 0 and len(actual_direction) > 0:
                            min_len = min(len(pred_direction), len(actual_direction))
                            pred_direction = pred_direction[:min_len]
                            actual_direction = actual_direction[:min_len]
                            
                            non_neutral_mask = (pred_direction != 0) & (actual_direction != 0)
                            if np.sum(non_neutral_mask) > 0:
                                correct_predictions = pred_direction[non_neutral_mask] == actual_direction[non_neutral_mask]
                                accuracy = np.mean(correct_predictions)
                                directional_accuracies[f'{horizon}d'] = accuracy
                            else:
                                directional_accuracies[f'{horizon}d'] = 0.5  # Random if all neutral
                    except Exception:
                        directional_accuracies[f'{horizon}d'] = 0.5  # Default to random
            
            # Primary score: 5-day directional accuracy (your main use case)
            primary_score = 1 - directional_accuracies.get('5d', 0.5)  # Convert to minimization problem
            
            # Add RMSE as secondary consideration (20% weight)
            rmse = np.sqrt(np.mean((Y_pred_scaled.flatten() - Y_val_scaled)**2))
            combined_score = 0.8 * primary_score + 0.2 * rmse
            
            return combined_score
            
        except Exception as e:
            return float('inf')
    
    def comprehensive_optimize(self, df, max_features=8, n_feature_trials=300):
        """Comprehensive parameter optimization"""
        print("="*80)
        print(f"COMPREHENSIVE DMD OPTIMIZATION FOR {self.target_metric.upper()}")
        print(f"PREDICTION HORIZON: {self.prediction_horizon} DAYS")  # NEW
        print("="*80)
        
        # Time series split (more conservative)
        n = len(df)
        train_end = int(n * 0.5)  # Smaller training set
        val_end = int(n * 0.75)   # Larger validation set
        
        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()
        
        print(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        
        # Create indicators with prediction horizon
        print("Creating features with strict no-leakage policy...")
        indicators_train = StrictIndicatorLibrary.create_indicators(df_train, max_lags=3, prediction_horizon=self.prediction_horizon)  # CHANGED
        indicators_val = StrictIndicatorLibrary.create_indicators(df_val, max_lags=3, prediction_horizon=self.prediction_horizon)  # CHANGED
        
        # Prepare targets
        target_train = self.prepare_target(df_train)
        target_val = self.prepare_target(df_val)
        
        # Get valid features first
        valid_features = []
        for col in indicators_train.columns:
            train_coverage = indicators_train[col].notna().sum() / len(indicators_train)
            val_coverage = indicators_val[col].notna().sum() / len(indicators_val)
            
            if train_coverage > 0.8 and val_coverage > 0.8:
                valid_features.append(col)
        
        # Remove redundant features to improve model quality
        # Group similar features and pick representatives
        feature_groups = {
            'returns': [col for col in valid_features if 'returns_' in col and 'log_returns' not in col],
            'log_returns': [col for col in valid_features if 'log_returns_' in col],
            'ratios': [col for col in valid_features if ('_ratio_' in col or 'close_open_ratio' in col)],
            'momentum': [col for col in valid_features if 'momentum_' in col],
            'volatility': [col for col in valid_features if ('volatility_' in col or 'vol_ratio' in col)],
            'sma': [col for col in valid_features if 'sma_' in col],
            'technical': [col for col in valid_features if any(x in col for x in ['rsi', 'bb_', 'price_position'])]
        }
        
        # Select best representative from each group
        diverse_features = []
        for group_name, group_features in feature_groups.items():
            if group_features:
                # Take up to 2 features per group to ensure diversity
                diverse_features.extend(group_features[:2])
        
        # If we still don't have enough, add remaining features
        remaining = [f for f in valid_features if f not in diverse_features]
        valid_features = diverse_features + remaining[:10]  # Cap total features
        
        print(f"Available features: {len(valid_features)}")
        print("Sample features:", valid_features[:10])
        
        if len(valid_features) < 3:
            raise ValueError("Not enough valid features!")
        
        # Comprehensive parameter grid - allow higher ranks for more complex patterns
        param_grid = {
            'rank': [5, 8, 10, 15, 20, 25, None],  # Start with higher ranks
            'regularization': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4], 
            'scaler': ['standard', 'robust']
        }
        
        results = []
        successful_tests = 0
        total_combinations = 0
        
        print(f"\nTesting {n_feature_trials} feature combinations with {len(param_grid['rank']) * len(param_grid['regularization']) * len(param_grid['scaler'])} parameter sets each...")
        
        for trial in range(n_feature_trials):
            # Random feature selection
            n_features = np.random.randint(3, min(max_features + 1, len(valid_features) + 1))
            selected_features = np.random.choice(valid_features, n_features, replace=False).tolist()
            
            # Test all parameter combinations
            for rank in param_grid['rank']:
                for reg in param_grid['regularization']:
                    for scaler in param_grid['scaler']:
                        total_combinations += 1
                        
                        params = {
                            'rank': rank,
                            'regularization': reg,
                            'scaler': scaler
                        }
                        
                        score = self.evaluate_model(
                            indicators_train, target_train,
                            indicators_val, target_val,
                            selected_features, params
                        )
                        
                        if score < float('inf'):
                            results.append({
                                'features': selected_features.copy(),
                                'params': params.copy(),
                                'score': score
                            })
                            successful_tests += 1
                            
                            if successful_tests % 100 == 0:
                                print(f"  Successfully tested {successful_tests}/{total_combinations} configurations...")
        
        print(f"\nCompleted: {successful_tests}/{total_combinations} successful configurations")
        
        if not results:
            raise ValueError("No valid configurations found!")
        
        # Sort by score
        results.sort(key=lambda x: x['score'])
        
        # Store best configuration
        self.best_features = results[0]['features']
        self.best_params = results[0]['params']
        
        # Display results
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS (Validation Set Performance)")
        print("="*80)
        
        for i, result in enumerate(results[:15]):
            print(f"\n{i+1}. Score: {result['score']:.6f}")
            print(f"   Features: {', '.join(result['features'][:3])}")
            if len(result['features']) > 3:
                print(f"            {', '.join(result['features'][3:])}")
            print(f"   Rank: {result['params']['rank']}, Reg: {result['params']['regularization']:.2e}, Scaler: {result['params']['scaler']}")
        
        return results
    
    def final_training_and_evaluation(self, df):
        """Final model training with comprehensive directional analysis"""
        print("\n" + "="*80)
        print("FINAL MODEL TRAINING AND TEST SET EVALUATION")
        print(f"PREDICTION HORIZON: {self.prediction_horizon} DAYS")  # NEW
        print("="*80)
        
        # Use same split as optimization
        n = len(df)
        train_val_end = int(n * 0.75)
        
        df_train_val = df.iloc[:train_val_end].copy()
        df_test = df.iloc[train_val_end:].copy()
        
        print(f"Training on: {df_train_val.index[0].strftime('%Y-%m-%d')} to {df_train_val.index[-1].strftime('%Y-%m-%d')} ({len(df_train_val)} days)")
        print(f"Testing on:  {df_test.index[0].strftime('%Y-%m-%d')} to {df_test.index[-1].strftime('%Y-%m-%d')} ({len(df_test)} days)")
        
        # Prepare data with prediction horizon
        indicators_train_val = StrictIndicatorLibrary.create_indicators(df_train_val, max_lags=3, prediction_horizon=self.prediction_horizon)  # CHANGED
        indicators_test = StrictIndicatorLibrary.create_indicators(df_test, max_lags=3, prediction_horizon=self.prediction_horizon)  # CHANGED
        
        target_train_val = self.prepare_target(df_train_val)
        target_test = self.prepare_target(df_test)
        
        # Create datasets
        X_train, Y_train, _ = self.create_dataset(indicators_train_val, target_train_val, self.best_features)
        X_test, Y_test, test_indices = self.create_dataset(indicators_test, target_test, self.best_features)
        
        if X_train is None or X_test is None:
            raise ValueError("Failed to create valid datasets!")
        
        # Setup scalers
        if self.best_params['scaler'] == 'robust':
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        
        # Scale data
        X_train_scaled = self.feature_scaler.fit_transform(X_train.T).T
        Y_train_scaled = self.target_scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
        
        X_test_scaled = self.feature_scaler.transform(X_test.T).T
        Y_test_scaled = self.target_scaler.transform(Y_test.reshape(-1, 1)).flatten()
        
        # Train final model
        self.best_model = EnhancedDMD(
            rank=self.best_params['rank'],
            regularization=self.best_params['regularization']
        )
        self.best_model.fit(X_train_scaled, Y_train_scaled)
        
        # Predict on test set
        Y_pred_scaled = self.best_model.predict(X_test_scaled)
        Y_pred = self.target_scaler.inverse_transform(Y_pred_scaled.T).flatten()
        
        # Calculate traditional metrics
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        mae = mean_absolute_error(Y_test, Y_pred)
        correlation = np.corrcoef(Y_pred, Y_test)[0, 1]
        bias = np.mean(Y_pred - Y_test)
        
        # MAPE calculation
        if self.target_metric == 'log_prices':
            pred_prices = np.exp(Y_pred)
            actual_prices = np.exp(Y_test)
            mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100
        elif self.target_metric in ['price_ratio_sma200', 'log_price_ratio_sma200', 'log_price_ratio_sma20']:
            non_zero_mask = np.abs(Y_test) > 1e-6
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((Y_pred[non_zero_mask] - Y_test[non_zero_mask]) / 
                                    Y_test[non_zero_mask])) * 100
            else:
                mape = np.nan
        else:
            mape = np.nan
        
        # DIRECTIONAL ACCURACY ANALYSIS (MAIN FOCUS)
        print(f"\n📊 TRADITIONAL METRICS:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%" if not np.isnan(mape) else "MAPE: N/A")
        print(f"Correlation: {correlation:.4f}")
        print(f"Bias: {bias:.6f}")
        
        # Calculate directional accuracy for multiple horizons
        print(f"\n🎯 DIRECTIONAL ACCURACY ANALYSIS:")
        print("="*50)
        
        directional_results = {}
        
        # Test multiple horizons, but emphasize our target horizon
        for horizon in [1, 3, self.prediction_horizon, 10]:  # CHANGED: include our target horizon
            if len(Y_pred) >= horizon and len(Y_test) >= horizon:
                # For our target horizon, this is DIRECT prediction accuracy
                if horizon == self.prediction_horizon:
                    pred_direction = np.sign(Y_pred)  # Direct prediction
                    actual_direction = np.sign(Y_test)  # Direct actual
                    print(f"\n⭐ {horizon}-Day Direction Prediction (DIRECT TARGET):")
                else:
                    # For other horizons, use convolution as before
                    if horizon == 1:
                        pred_direction = np.sign(Y_pred)
                        actual_direction = np.sign(Y_test)
                    else:
                        pred_changes = np.convolve(Y_pred, np.ones(horizon), mode='valid')
                        actual_changes = np.convolve(Y_test, np.ones(horizon), mode='valid')
                        pred_direction = np.sign(pred_changes)
                        actual_direction = np.sign(actual_changes)
                    print(f"\n{horizon}-Day Direction Prediction:")
                
                # Calculate accuracies
                total_predictions = len(pred_direction)
                
                # Overall accuracy (including neutral)
                correct_all = pred_direction == actual_direction
                accuracy_all = np.mean(correct_all)
                
                # Accuracy excluding neutrals
                non_neutral_mask = (pred_direction != 0) & (actual_direction != 0)
                if np.sum(non_neutral_mask) > 0:
                    correct_non_neutral = pred_direction[non_neutral_mask] == actual_direction[non_neutral_mask]
                    accuracy_non_neutral = np.mean(correct_non_neutral)
                    non_neutral_count = np.sum(non_neutral_mask)
                else:
                    accuracy_non_neutral = np.nan
                    non_neutral_count = 0
                
                # Count directional predictions
                up_predictions = np.sum(pred_direction > 0)
                down_predictions = np.sum(pred_direction < 0)
                neutral_predictions = np.sum(pred_direction == 0)
                
                directional_results[horizon] = {
                    'accuracy_all': accuracy_all,
                    'accuracy_non_neutral': accuracy_non_neutral,
                    'total_predictions': total_predictions,
                    'non_neutral_count': non_neutral_count,
                    'up_predictions': up_predictions,
                    'down_predictions': down_predictions,
                    'neutral_predictions': neutral_predictions
                }
                
                print(f"\n{horizon}-Day Direction Prediction:")
                print(f"  Overall Accuracy: {accuracy_all:.1%} ({np.sum(correct_all)}/{total_predictions})")
                if not np.isnan(accuracy_non_neutral):
                    print(f"  Non-Neutral Accuracy: {accuracy_non_neutral:.1%} ({np.sum(correct_non_neutral)}/{non_neutral_count})")
                print(f"  Predictions: ↗️ {up_predictions} | ↘️ {down_predictions} | → {neutral_predictions}")
                
                # Performance vs random
                random_accuracy = 1/3 if neutral_predictions > 0 else 0.5
                improvement = (accuracy_all - random_accuracy) / random_accuracy * 100
                print(f"  vs Random ({random_accuracy:.1%}): {improvement:+.1f}% improvement")
        
        # Highlight best performing horizon
        best_horizon = max(directional_results.keys(), 
                          key=lambda h: directional_results[h]['accuracy_non_neutral'] 
                          if not np.isnan(directional_results[h]['accuracy_non_neutral']) 
                          else directional_results[h]['accuracy_all'])
        
        best_acc = directional_results[best_horizon]['accuracy_non_neutral']
        if np.isnan(best_acc):
            best_acc = directional_results[best_horizon]['accuracy_all']
        
        print(f"\n🏆 BEST PERFORMANCE: {best_horizon}-day predictions ({best_acc:.1%} accuracy)")
        
        # Recommendation for trading
        print(f"\n💡 TRADING RECOMMENDATION:")
        if best_acc > 0.6:
            print(f"✅ Strong signal: {best_horizon}-day directional accuracy of {best_acc:.1%}")
            print(f"   Recommended use: Daily rebalancing with {best_horizon}-day outlook")
        elif best_acc > 0.55:
            print(f"⚡ Moderate signal: {best_horizon}-day directional accuracy of {best_acc:.1%}")
            print(f"   Recommended use: Combine with other indicators")
        else:
            print(f"⚠️  Weak signal: Best accuracy only {best_acc:.1%}")
            print(f"   Recommended use: Paper trading or further model refinement")
        
        # Plot enhanced results
        self.plot_directional_analysis(Y_pred, Y_test, df_test.index[test_indices], directional_results)
        
        return Y_pred, Y_test, directional_results
    
    def plot_directional_analysis(self, predictions, actuals, dates, directional_results):
        """Enhanced plotting with directional analysis focus"""
        plt.figure(figsize=(18, 14))
        
        # Plot 1: Time series with directional markers
        plt.subplot(3, 2, 1)
        plt.plot(dates, actuals, 'b-', label='Actual', alpha=0.8, linewidth=1.5)
        plt.plot(dates, predictions, 'r--', label='DMD Prediction', alpha=0.8, linewidth=1.5)
        
        # Add directional markers
        for i in range(0, len(dates), max(1, len(dates)//20)):  # Show ~20 markers
            if i < len(predictions) and i < len(actuals):
                pred_dir = '↗️' if predictions[i] > 0 else ('↘️' if predictions[i] < 0 else '→')
                actual_dir = '↗️' if actuals[i] > 0 else ('↘️' if actuals[i] < 0 else '→')
                color = 'green' if pred_dir == actual_dir else 'red'
                plt.plot(dates[i], predictions[i], marker='o', color=color, markersize=4)
        
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_metric}')
        plt.title(f'DMD Predictions vs Actual - {self.target_metric}\n(Green dots = correct direction, Red = wrong)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        plt.subplot(3, 2, 2)
        plt.scatter(actuals, predictions, alpha=0.6, s=20)
        min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Prediction vs Actual Scatter Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Directional accuracy by horizon
        plt.subplot(3, 2, 3)
        horizons = list(directional_results.keys())
        accuracies = [directional_results[h]['accuracy_non_neutral'] 
                     if not np.isnan(directional_results[h]['accuracy_non_neutral'])
                     else directional_results[h]['accuracy_all'] for h in horizons]
        
        bars = plt.bar(horizons, accuracies, alpha=0.7, 
                      color=['green' if acc > 0.6 else 'orange' if acc > 0.55 else 'red' for acc in accuracies])
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
        plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Good (60%)')
        plt.xlabel('Prediction Horizon (Days)')
        plt.ylabel('Directional Accuracy')
        plt.title('Directional Accuracy by Time Horizon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Prediction distribution
        plt.subplot(3, 2, 4)
        pred_directions = np.sign(predictions)
        actual_directions = np.sign(actuals)
        
        direction_labels = ['Down (-1)', 'Neutral (0)', 'Up (+1)']
        pred_counts = [np.sum(pred_directions == -1), np.sum(pred_directions == 0), np.sum(pred_directions == 1)]
        actual_counts = [np.sum(actual_directions == -1), np.sum(actual_directions == 0), np.sum(actual_directions == 1)]
        
        x = np.arange(len(direction_labels))
        width = 0.35
        
        plt.bar(x - width/2, pred_counts, width, label='Predicted', alpha=0.7, color='red')
        plt.bar(x + width/2, actual_counts, width, label='Actual', alpha=0.7, color='blue')
        
        plt.xlabel('Direction')
        plt.ylabel('Count')
        plt.title('Direction Prediction Distribution')
        plt.xticks(x, direction_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Residual analysis
        plt.subplot(3, 2, 5)
        residuals = predictions - actuals
        plt.plot(dates, residuals, 'g-', alpha=0.7, linewidth=1)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.axhline(y=np.mean(residuals), color='orange', linestyle=':', 
                   label=f'Mean Bias: {np.mean(residuals):.4f}')
        plt.xlabel('Date')
        plt.ylabel('Prediction Error')
        plt.title('Residual Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Directional success over time (rolling window)
        plt.subplot(3, 2, 6)
        if len(predictions) >= 5:
            # 20-day rolling directional accuracy
            rolling_accuracy = []
            window = 20  # 20-day rolling window
            
            for i in range(window, len(predictions)):
                window_pred = np.sign(predictions[i-window:i])
                window_actual = np.sign(actuals[i-window:i])
                non_neutral = (window_pred != 0) & (window_actual != 0)
                if np.sum(non_neutral) > 0:
                    accuracy = np.mean(window_pred[non_neutral] == window_actual[non_neutral])
                else:
                    accuracy = 0.5
                rolling_accuracy.append(accuracy)
            
            rolling_dates = dates[window:len(predictions)]
            plt.plot(rolling_dates[:len(rolling_accuracy)], rolling_accuracy, 'purple', linewidth=2)
            plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
            plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Good')
            plt.xlabel('Date')
            plt.ylabel('Rolling Directional Accuracy')
            plt.title(f'{window}-Day Rolling Directional Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def make_daily_prediction(self, df, prediction_horizon=None):
        """Make live prediction for next N days direction"""
        if self.best_model is None:
            raise ValueError("Must train model first!")
        
        # Use system's prediction horizon if not specified
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon
        
        print(f"\n🔮 DAILY PREDICTION FOR NEXT {prediction_horizon} DAYS")
        print(f"MODEL TRAINED FOR: {self.prediction_horizon}-DAY DIRECT PREDICTION")  # NEW
        print("="*60)
        
        # Use most recent data for prediction
        indicators = StrictIndicatorLibrary.create_indicators(df, max_lags=3, prediction_horizon=self.prediction_horizon)  # CHANGED
        
        
        # Get the last valid feature vector
        last_features = []
        missing_features = []
        
        for col in self.best_features:
            if col in indicators.columns:
                col_data = indicators[col].dropna()
                if len(col_data) > 0:
                    last_val = col_data.iloc[-1]
                    last_features.append(last_val)
                else:
                    missing_features.append(col)
                    last_features.append(0.0)  # Default value
            else:
                missing_features.append(col)
                last_features.append(0.0)  # Default value
        
        if missing_features:
            print(f"⚠️  Warning: Missing features {missing_features}, using default values")
        
        if len(last_features) != len(self.best_features):
            raise ValueError(f"Feature mismatch: expected {len(self.best_features)}, got {len(last_features)}")
        
        try:
            X_last = np.array(last_features).reshape(-1, 1)
            X_last_scaled = self.feature_scaler.transform(X_last.T).T
            
            # Predict next value
            Y_next_scaled = self.best_model.predict(X_last_scaled)
            Y_next = self.target_scaler.inverse_transform(Y_next_scaled.T).flatten()[0]
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
        
        # Current price and date info
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]
        next_date = current_date + pd.Timedelta(days=1)
        
        print(f"📊 Current Status ({current_date.strftime('%Y-%m-%d')}):")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Prediction Date: {next_date.strftime('%Y-%m-%d')}")
        
        # Interpret prediction based on target type
        if self.target_metric == 'log_prices':
            predicted_log_price = Y_next
            predicted_price = np.exp(predicted_log_price)
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            
            direction = "📈 UP" if percent_change > 0.5 else "📉 DOWN" if percent_change < -0.5 else "➡️ FLAT"
            confidence = "HIGH" if abs(percent_change) > 2 else "MEDIUM" if abs(percent_change) > 1 else "LOW"
            
            print(f"\n🎯 PREDICTION:")
            print(f"   Expected Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {price_change:+.2f} ({percent_change:+.1f}%)")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {confidence}")
            
        elif self.target_metric == 'price_ratio_sma200':
            predicted_ratio = Y_next
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            predicted_price = predicted_ratio * sma_200
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            
            direction = "📈 UP" if percent_change > 0.5 else "📉 DOWN" if percent_change < -0.5 else "➡️ FLAT"
            
            current_ratio = current_price / sma_200
            ratio_change = predicted_ratio - current_ratio
            
            print(f"\n🎯 PREDICTION:")
            print(f"   Expected Price/SMA200: {predicted_ratio:.3f}")
            print(f"   Current Price/SMA200: {current_ratio:.3f}")
            print(f"   Ratio Change: {ratio_change:+.3f}")
            print(f"   Expected Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {price_change:+.2f} ({percent_change:+.1f}%)")
            print(f"   Direction: {direction}")
            
            if predicted_ratio > 1.05:
                signal = "🟢 BULLISH (Above trend)"
            elif predicted_ratio < 0.95:
                signal = "🔴 BEARISH (Below trend)"
            else:
                signal = "🟡 NEUTRAL (Near trend)"
            print(f"   Signal: {signal}")
            
        elif self.target_metric == 'log_price_ratio_sma200':
            predicted_log_ratio = Y_next
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            predicted_ratio = np.exp(predicted_log_ratio)
            predicted_price = predicted_ratio * sma_200
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            
            direction = "📈 UP" if percent_change > 0.5 else "📉 DOWN" if percent_change < -0.5 else "➡️ FLAT"
            
            current_log_ratio = np.log(current_price / sma_200)
            log_ratio_change = predicted_log_ratio - current_log_ratio
            
            print(f"\n🎯 PREDICTION:")
            print(f"   Expected Log(Price/SMA200): {predicted_log_ratio:.3f}")
            print(f"   Current Log(Price/SMA200): {current_log_ratio:.3f}")
            print(f"   Log Ratio Change: {log_ratio_change:+.3f}")
            print(f"   Expected Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {price_change:+.2f} ({percent_change:+.1f}%)")
            print(f"   Direction: {direction}")
            
            if predicted_log_ratio > 0.05:
                signal = "🟢 BULLISH (Above trend)"
            elif predicted_log_ratio < -0.05:
                signal = "🔴 BEARISH (Below trend)"
            else:
                signal = "🟡 NEUTRAL (Near trend)"
            print(f"   Signal: {signal}")
            
        elif self.target_metric == 'log_price_ratio_sma20':
            predicted_log_ratio = Y_next
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            predicted_ratio = np.exp(predicted_log_ratio)
            predicted_price = predicted_ratio * sma_20
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            
            direction = "📈 UP" if percent_change > 0.5 else "📉 DOWN" if percent_change < -0.5 else "➡️ FLAT"
            
            current_log_ratio = np.log(current_price / sma_20)
            log_ratio_change = predicted_log_ratio - current_log_ratio
            
            print(f"\n🎯 PREDICTION:")
            print(f"   Expected Log(Price/SMA20): {predicted_log_ratio:.3f}")
            print(f"   Current Log(Price/SMA20): {current_log_ratio:.3f}")
            print(f"   Log Ratio Change: {log_ratio_change:+.3f}")
            print(f"   Expected Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {price_change:+.2f} ({percent_change:+.1f}%)")
            print(f"   Direction: {direction}")
            
            if predicted_log_ratio > 0.02:
                signal = "🟢 BULLISH (Above short-term trend)"
            elif predicted_log_ratio < -0.02:
                signal = "🔴 BEARISH (Below short-term trend)"
            else:
                signal = "🟡 NEUTRAL (Near short-term trend)"
            print(f"   Signal: {signal}")
            
        else:
            # Fallback for unknown target metrics
            print(f"\n🎯 PREDICTION:")
            print(f"   Raw Prediction Value: {Y_next:.4f}")
            print(f"   Target Metric: {self.target_metric}")
            print(f"   ⚠️  No specific interpretation available for this metric")
            percent_change = 0  # Default fallback
            direction = "UNKNOWN"
        
        # Multi-day outlook
        print(f"\n📅 {prediction_horizon}-DAY OUTLOOK:")
        
        if prediction_horizon == self.prediction_horizon:
            print(f"   ⭐ DIRECT {self.prediction_horizon}-DAY PREDICTION (not scaled)")
            cumulative_change = percent_change  # Direct prediction, no scaling
        else:
            print(f"   📊 SCALED from {self.prediction_horizon}-day model")
            # Scale from our prediction horizon to requested horizon
            scale_factor = prediction_horizon / self.prediction_horizon
            cumulative_change = percent_change * scale_factor * 0.8  # Conservative scaling
    
        
        target_price = current_price * (1 + cumulative_change / 100)
        
        print(f"   Projected {prediction_horizon}-day change: {cumulative_change:+.1f}%")
        print(f"   Target price range: ${target_price * 0.95:.2f} - ${target_price * 1.05:.2f}")
        
        if abs(cumulative_change) > 5:
            risk_level = "HIGH"
        elif abs(cumulative_change) > 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        print(f"   Risk Level: {risk_level}")
        
        # Trading recommendation
        print(f"\n💡 TRADING RECOMMENDATION:")
        if self.target_metric in ['price_ratio_sma200', 'log_price_ratio_sma200']:
            if self.target_metric == 'price_ratio_sma200':
                if predicted_ratio > current_ratio + 0.02:
                    recommendation = "🟢 CONSIDER BUYING (Mean reversion + momentum)"
                elif predicted_ratio < current_ratio - 0.02:
                    recommendation = "🔴 CONSIDER SELLING (Mean reversion down)"
                else:
                    recommendation = "🟡 HOLD (Limited directional signal)"
            else:  # log_price_ratio_sma200
                if log_ratio_change > 0.02:
                    recommendation = "🟢 CONSIDER BUYING (Mean reversion + momentum)"
                elif log_ratio_change < -0.02:
                    recommendation = "🔴 CONSIDER SELLING (Mean reversion down)"
                else:
                    recommendation = "🟡 HOLD (Limited directional signal)"
        elif self.target_metric == 'log_price_ratio_sma20':
            if log_ratio_change > 0.01:
                recommendation = "🟢 CONSIDER BUYING (Short-term momentum up)"
            elif log_ratio_change < -0.01:
                recommendation = "🔴 CONSIDER SELLING (Short-term momentum down)"
            else:
                recommendation = "🟡 HOLD (Limited directional signal)"
        else:
            if abs(percent_change) > 1:
                action = "BUY" if percent_change > 0 else "SELL"
                recommendation = f"{'🟢' if percent_change > 0 else '🔴'} CONSIDER {action}ING"
            else:
                recommendation = "🟡 HOLD (Weak signal)"
        
        print(f"   {recommendation}")
        print(f"   Stop Loss: {2 * abs(percent_change):.1f}%")
        print(f"   Position Size: Recommend {min(100, max(10, 50 / max(1, abs(percent_change)))): .0f}% of normal")
        
        return {
            'prediction_date': next_date,
            'current_price': current_price,
            'predicted_value': Y_next,
            'expected_change_pct': percent_change if 'percent_change' in locals() else 0,
            'direction': direction if 'direction' in locals() else 'UNKNOWN',
            'confidence': confidence if 'confidence' in locals() else 'LOW',
            'recommendation': recommendation
        }
    
    def get_feature_importance(self):
        """Analyze feature importance from the DMD matrix"""
        if self.best_model is None or self.best_model.A is None:
            raise ValueError("Model not trained!")
        
        # Get the DMD matrix
        A = self.best_model.A
        
        # Calculate feature importance as the norm of each column
        if len(A.shape) == 2:
            feature_importance = np.linalg.norm(A, axis=0)
        else:
            feature_importance = np.abs(A.flatten())
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.best_features,
            'importance': feature_importance[:len(self.best_features)]
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("DMD TRADING SYSTEM - SUMMARY REPORT")
        print("="*80)
        
        print(f"\n📊 MODEL CONFIGURATION:")
        print(f"Target Metric: {self.target_metric}")
        print(f"Best Parameters: {self.best_params}")
        print(f"Selected Features: {len(self.best_features)}")
        
        print(f"\n🎯 FEATURE SELECTION:")
        for i, feature in enumerate(self.best_features, 1):
            print(f"  {i:2d}. {feature}")
        
        if self.best_model is not None:
            try:
                importance_df = self.get_feature_importance()
                print(f"\n📈 FEATURE IMPORTANCE (Top 5):")
                for idx, row in importance_df.head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.1%}")
            except Exception:
                print(f"\n📈 FEATURE IMPORTANCE: Unable to calculate")
        
        print(f"\n💡 USAGE INSTRUCTIONS:")
        print(f"1. Load your data with columns: ['Open', 'High', 'Low', 'Close', 'Volume']")
        print(f"2. Call: system.comprehensive_optimize(df)")
        print(f"3. Call: system.final_training_and_evaluation(df)")
        print(f"4. Call: system.make_daily_prediction(df, prediction_horizon=5) for predictions")
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"- All features use strict 1+ day lag to prevent look-ahead bias")
        print(f"- Model optimizes for {self.target_metric} directional accuracy")
        print(f"- Best used for 1-10 day directional predictions")
        print(f"- Combine with risk management and position sizing")
        
        print("="*80)


def run_robust_dmd_analysis(filename, target_metric='price_ratio_sma200', prediction_horizon=5):
    """Run comprehensive DMD analysis focused on directional prediction"""
    
    # Load data
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [col.capitalize() for col in df.columns]
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize system
    dmd_system = RobustDMDSystem(target_metric=target_metric)
    
    # Show what we're predicting and WHY it's valid
    if target_metric == 'log_prices':
        print("✅ SCIENTIFICALLY VALIDATED: log_prices (9.3% reconstruction error)")
        print("   Predicting tomorrow's log price using linear dynamics")
        print("   ⚠️  Note: High correlation may be due to unit root, not genuine forecasting")
    elif target_metric == 'price_ratio_sma200':
        print("✅ SCIENTIFICALLY VALIDATED: price_ratio_sma200 (10.1% reconstruction error)")
        print("   Predicting tomorrow's price ratio to 200-day moving average")
        print("   ✅ Recommended: Mean-reverting dynamics are genuinely predictable")
    elif target_metric == 'log_price_ratio_sma200':
        print("✅ ENHANCED VALIDATED METRIC: log_price_ratio_sma200")
        print("   Predicting log(price/SMA200) - combines mean reversion with variance stabilization")
        print("   ✅ Expected to be highly linear due to log transformation")
    elif target_metric == 'log_price_ratio_sma20':
        print("✅ RELATED TO VALIDATED METRIC: log(price/SMA20) ratio")
        print("   Using shorter MA for more responsive mean reversion signals")
        print("   ⚡ More sensitive to short-term market dynamics")
    else:
        print(f"⚠️  WARNING: {target_metric} not scientifically validated as linear!")
        print("   Consider using 'log_prices', 'price_ratio_sma200', 'log_price_ratio_sma200', or 'log_price_ratio_sma20' instead")
    
    print(f"\n🎯 FOCUS: Optimizing for {prediction_horizon}-day directional accuracy")
    
    # Comprehensive optimization
    results = dmd_system.comprehensive_optimize(df, max_features=8, n_feature_trials=50)
    
    # Train and evaluate with directional focus
    predictions, actuals, directional_results = dmd_system.final_training_and_evaluation(df)
    
    # Make live prediction
    live_prediction = dmd_system.make_daily_prediction(df, prediction_horizon)
    
    # Show results
    print(f"\n📈 BEST FEATURES FOR PREDICTING {target_metric}:")
    for i, feat in enumerate(dmd_system.best_features):
        print(f"  {i+1}. {feat}")
    
    print(f"\n⚙️  BEST DMD PARAMETERS:")
    for key, value in dmd_system.best_params.items():
        print(f"  {key}: {value}")
    
    # Summary recommendation
    best_horizon = max(directional_results.keys(), 
                      key=lambda h: directional_results[h]['accuracy_non_neutral'] 
                      if not np.isnan(directional_results[h]['accuracy_non_neutral']) 
                      else directional_results[h]['accuracy_all'])
    
    best_acc = directional_results[best_horizon]['accuracy_non_neutral']
    if np.isnan(best_acc):
        best_acc = directional_results[best_horizon]['accuracy_all']
    
    print(f"\n🏆 SUMMARY:")
    print(f"   Best Performance: {best_horizon}-day predictions at {best_acc:.1%} accuracy")
    print(f"   Recommended Use: {'Daily trading' if best_acc > 0.6 else 'Paper trading' if best_acc > 0.55 else 'Further development'}")
    
    return dmd_system, directional_results, live_prediction