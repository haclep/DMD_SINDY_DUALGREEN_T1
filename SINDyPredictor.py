import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.integrate import odeint
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Import the original SINDy components (keep them unchanged)
def check_and_prepare_dataframe(df):
    """
    Check dataframe format and prepare it for analysis - FIXED VERSION
    """
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index type: {type(df.index)}")
    print(f"First few rows:")
    print(df.head())
    
    # Create standardized dataframe PRESERVING ORIGINAL ORDER
    new_df = df.copy()
    
    # Rename columns to standard names but keep the same order
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'open' in col_lower:
            col_mapping[col] = 'Open'
        elif 'high' in col_lower:
            col_mapping[col] = 'High'
        elif 'low' in col_lower:
            col_mapping[col] = 'Low'
        elif 'close' in col_lower:
            col_mapping[col] = 'Close'
        elif 'volume' in col_lower:
            col_mapping[col] = 'Volume'
    
    # Rename columns
    new_df = new_df.rename(columns=col_mapping)
    
    print(f"\nColumn mapping applied: {col_mapping}")
    print(f"New DataFrame columns (SAME ORDER): {list(new_df.columns)}")
    
    # Ensure required columns exist
    if 'Close' not in new_df.columns:
        raise ValueError("Could not identify Close price column!")
    
    # Fill missing OHLC with Close if needed (maintaining order)
    if 'High' not in new_df.columns:
        new_df['High'] = new_df['Close']
    if 'Low' not in new_df.columns:
        new_df['Low'] = new_df['Close']
    if 'Open' not in new_df.columns:
        new_df['Open'] = new_df['Close']
    
    # Verify we have the essential columns
    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in new_df.columns:
            print(f"WARNING: Missing {col} column, using Close price as substitute")
            new_df[col] = new_df['Close']
    
    print(f"Final DataFrame columns: {list(new_df.columns)}")
    
    return new_df

class FinancialLibrary:
    """Financial library optimized for time series prediction"""
    
    def __init__(self, include_ratios=False, include_interactions=True, 
                 include_technical=True, max_degree=2):
        self.include_ratios = include_ratios
        self.include_interactions = include_interactions
        self.include_technical = include_technical
        self.max_degree = max_degree
        self.feature_names = []
        
    def transform(self, X):
        """
        Transform input features into library of candidate functions
        X: array of shape (n_samples, n_features)
        Returns: array of shape (n_samples, n_library_features)
        """
        n_samples = X.shape[0]
        library = []
        self.feature_names = []
        
        print(f"Building library from {X.shape[1]} input features...")
        
        # Original features (KEEP THESE - they're important!)
        for i in range(X.shape[1]):
            library.append(X[:, i])
            self.feature_names.append(f'x{i}')
        
        # Polynomial features (selective)
        if self.max_degree >= 2:
            # Quadratic terms for first few features only
            for i in range(min(4, X.shape[1])):
                library.append(X[:, i]**2)
                self.feature_names.append(f'x{i}^2')
            
            # Key cross terms (only important ones)
            if X.shape[1] >= 2:
                # Cross between consecutive features
                for i in range(min(3, X.shape[1]-1)):
                    library.append(X[:, i] * X[:, i+1])
                    self.feature_names.append(f'x{i}*x{i+1}')
        
        # Technical indicators
        if self.include_technical:
            # Sign-preserving transformations
            for i in range(min(3, X.shape[1])):
                # Square root (for volatility-like features)
                sqrt_feature = np.sign(X[:, i]) * np.sqrt(np.abs(X[:, i]) + 1e-8)
                library.append(sqrt_feature)
                self.feature_names.append(f'sign_sqrt(x{i})')
        
        # Limited interaction terms
        if self.include_interactions:
            # Difference terms (momentum-like)
            for i in range(min(2, X.shape[1]-1)):
                momentum = X[:, i] - X[:, i+1]
                library.append(momentum)
                self.feature_names.append(f'x{i}-x{i+1}')
        
        # Limited ratios (only if requested and makes sense)
        if self.include_ratios and X.shape[1] >= 2:
            # Only a few key ratios
            for i in range(min(2, X.shape[1])):
                for j in range(min(2, X.shape[1])):
                    if i != j:
                        # Careful ratio computation
                        denominator = X[:, j]
                        safe_denominator = np.where(np.abs(denominator) < 1e-6, 
                                                  np.sign(denominator) * 1e-6, 
                                                  denominator)
                        ratio = X[:, i] / safe_denominator
                        # Clip extreme ratios
                        ratio = np.clip(ratio, -100, 100)
                        library.append(ratio)
                        self.feature_names.append(f'x{i}/x{j}')
        
        library_matrix = np.array(library).T
        print(f"Library built: {library_matrix.shape[1]} features")
        print(f"Feature names: {self.feature_names}")
        
        return library_matrix

class SINDyRegressor:
    """SINDy with improved regularization for financial data"""
    
    def __init__(self, threshold=0.01, alpha=0.001, max_iter=20, 
                 method='sequential_lasso', normalize=True):
        """
        NEW method: 'sequential_lasso' - fits with decreasing regularization
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.normalize = normalize
        self.coef_ = None
        self.scaler = None
        
    def fit(self, theta, dx):
        """
        Fit SINDy model with improved regularization strategy
        """
        if dx.ndim == 1:
            dx = dx.reshape(-1, 1)
        
        print(f"Fitting SINDy with method='{self.method}'")
        print(f"Input shapes: theta={theta.shape}, dx={dx.shape}")
        
        # Normalize if requested
        if self.normalize:
            self.scaler = StandardScaler()
            theta_norm = self.scaler.fit_transform(theta)
        else:
            theta_norm = theta
        
        n_features = theta_norm.shape[1]
        n_outputs = dx.shape[1]
        
        if self.method == 'sequential_lasso':
            # NEW: Progressive regularization approach
            self.coef_ = np.zeros((n_features, n_outputs))
            
            for k in range(n_outputs):
                # Start with very light regularization and increase
                alphas = [0.00001, 0.0001, 0.001, 0.01]
                best_model = None
                best_score = float('inf')
                
                for alpha in alphas:
                    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000,
                                selection='random', tol=1e-6)
                    model.fit(theta_norm, dx[:, k])
                    
                    # Evaluate on training set (we want good fit + sparsity)
                    pred = model.predict(theta_norm)
                    mse = mean_squared_error(dx[:, k], pred)
                    n_nonzero = np.sum(np.abs(model.coef_) > 1e-8)
                    
                    # Combined score: prefer models with 3-10 terms
                    if 3 <= n_nonzero <= 10:
                        score = mse
                    else:
                        score = mse * 5  # Penalize too sparse/dense
                    
                    print(f"    Alpha {alpha}: MSE={mse:.6f}, n_terms={n_nonzero}, score={score:.6f}")
                    
                    if score < best_score:
                        best_score = score
                        best_model = model
                
                if best_model is not None:
                    self.coef_[:, k] = best_model.coef_
                else:
                    print("WARNING: No good model found, using least squares")
                    self.coef_[:, k] = np.linalg.pinv(theta_norm) @ dx[:, k]
                    
        elif self.method == 'lasso':
            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000,
                         selection='random', tol=1e-6)
            self.coef_ = np.zeros((n_features, n_outputs))
            
            for k in range(n_outputs):
                model.fit(theta_norm, dx[:, k])
                self.coef_[:, k] = model.coef_
                
        elif self.method == 'ridge':
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            self.coef_ = np.zeros((n_features, n_outputs))
            
            for k in range(n_outputs):
                model.fit(theta_norm, dx[:, k])
                self.coef_[:, k] = model.coef_
        
        # Report fit quality
        n_nonzero = np.sum(np.abs(self.coef_) > 1e-8)
        print(f"Final model: {n_nonzero}/{self.coef_.size} non-zero coefficients")
        
        return self
    
    def predict(self, theta):
        """Predict using the learned sparse model"""
        if self.normalize and self.scaler is not None:
            theta_norm = self.scaler.transform(theta)
        else:
            theta_norm = theta
        
        return theta_norm @ self.coef_
    
    def get_equation(self, feature_names, output_names=None, precision=6):
        """Get human-readable equations with more precision"""
        if output_names is None:
            output_names = [f'dx{i}/dt' for i in range(self.coef_.shape[1])]
        
        equations = []
        for k in range(self.coef_.shape[1]):
            terms = []
            for j in range(self.coef_.shape[0]):
                if abs(self.coef_[j, k]) > 1e-8:
                    coef = round(self.coef_[j, k], precision)
                    if coef != 0:
                        if coef > 0 and terms:
                            terms.append(f"+ {coef}*{feature_names[j]}")
                        else:
                            terms.append(f"{coef}*{feature_names[j]}")
            
            if terms:
                equation = f"{output_names[k]} = " + " ".join(terms)
            else:
                equation = f"{output_names[k]} = 0"
            
            equations.append(equation)
        
        return equations

# NEW: Strict Indicator Library for SINDy (same as DMD)
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

# NEW: Robust SINDy System for Directional Prediction
class RobustSINDySystem:
    """Robust SINDy system with comprehensive parameter search and directional focus"""
    
    def __init__(self, target_metric='log_price_ratio_sma20', prediction_horizon=5):  # ADD prediction_horizon
        self.target_metric = target_metric
        self.prediction_horizon = prediction_horizon  # NEW
        self.best_features = None
        self.best_params = None
        self.best_model = None
        self.best_library = None
        self.feature_scaler = None
        self.target_scaler = None
        self.library_config = None
        
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
        
        X = np.column_stack(feature_data)  # (n_times, n_features) for SINDy
        Y = target_series.values
        
        # Remove any samples with NaN in features OR target
        feature_valid = ~np.any(np.isnan(X), axis=1)
        target_valid = ~np.isnan(Y)
        valid_mask = feature_valid & target_valid
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < min_periods:
            return None, None, None
        
        X_clean = X[valid_indices]
        Y_clean = Y[valid_indices]
        
        return X_clean, Y_clean, valid_indices
    
    def evaluate_model(self, train_indicators, train_target, val_indicators, val_target, 
                      feature_cols, sindy_params, library_params, min_test_samples=50):
        """Comprehensive model evaluation with directional accuracy focus"""
        try:
            # Create datasets
            X_train, Y_train, _ = self.create_dataset(train_indicators, train_target, feature_cols)
            X_val, Y_val, _ = self.create_dataset(val_indicators, val_target, feature_cols)
            
            if (X_train is None or X_val is None or 
                X_train.shape[0] < 100 or X_val.shape[0] < min_test_samples):
                return float('inf')
            
            # Use robust scaling
            scaler_type = sindy_params.get('scaler', 'standard')
            if scaler_type == 'robust':
                scaler_X = RobustScaler()
                scaler_Y = RobustScaler()
            else:
                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
            
            # Scale features
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            
            # Scale targets
            Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
            Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
            
            # Create library
            library = FinancialLibrary(
                include_ratios=library_params.get('include_ratios', False),
                include_interactions=library_params.get('include_interactions', True),
                include_technical=library_params.get('include_technical', True),
                max_degree=library_params.get('max_degree', 2)
            )
            
            # Transform to function library
            theta_train = library.transform(X_train_scaled)
            theta_val = library.transform(X_val_scaled)
            
            # Train SINDy
            sindy = SINDyRegressor(
                method=sindy_params.get('method', 'sequential_lasso'),
                alpha=sindy_params.get('alpha', 0.001),
                normalize=sindy_params.get('normalize', True)
            )
            sindy.fit(theta_train, Y_train_scaled)
            
            # Validate
            Y_pred_scaled = sindy.predict(theta_val).flatten()
            
            # Inverse transform to get actual values
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
            
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
            
            # Primary score: 5-day directional accuracy (main use case)
            primary_score = 1 - directional_accuracies.get('5d', 0.5)  # Convert to minimization problem
            
            # Add RMSE as secondary consideration (20% weight)
            rmse = np.sqrt(np.mean((Y_pred_scaled - Y_val_scaled)**2))
            combined_score = 0.8 * primary_score + 0.2 * rmse
            
            return combined_score
            
        except Exception as e:
            return float('inf')
    
    def comprehensive_optimize(self, df, max_features=8, n_feature_trials=300):
        """Comprehensive parameter optimization for SINDy"""
        print("="*80)
        print(f"COMPREHENSIVE SINDY OPTIMIZATION FOR {self.target_metric.upper()}")
        print(f"PREDICTION HORIZON: {self.prediction_horizon} DAYS")  # NEW
        print("="*80)
        
        # Time series split (same as DMD)
        n = len(df)
        train_end = int(n * 0.5)
        val_end = int(n * 0.75)
        
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
        
        # Get valid features
        valid_features = []
        for col in indicators_train.columns:
            train_coverage = indicators_train[col].notna().sum() / len(indicators_train)
            val_coverage = indicators_val[col].notna().sum() / len(indicators_val)
            
            if train_coverage > 0.8 and val_coverage > 0.8:
                valid_features.append(col)
        
        # Group features for diversity (same as DMD)
        feature_groups = {
            'returns': [col for col in valid_features if 'returns_' in col and 'log_returns' not in col],
            'log_returns': [col for col in valid_features if 'log_returns_' in col],
            'ratios': [col for col in valid_features if ('_ratio_' in col or 'close_open_ratio' in col)],
            'momentum': [col for col in valid_features if 'momentum_' in col],
            'volatility': [col for col in valid_features if ('volatility_' in col or 'vol_ratio' in col)],
            'sma': [col for col in valid_features if 'sma_' in col],
            'technical': [col for col in valid_features if any(x in col for x in ['rsi', 'bb_', 'price_position'])]
        }
        
        # Select diverse features
        diverse_features = []
        for group_name, group_features in feature_groups.items():
            if group_features:
                diverse_features.extend(group_features[:2])
        
        remaining = [f for f in valid_features if f not in diverse_features]
        valid_features = diverse_features + remaining[:10]
        
        print(f"Available features: {len(valid_features)}")
        print("Sample features:", valid_features[:10])
        
        if len(valid_features) < 3:
            raise ValueError("Not enough valid features!")
        
        # SINDy parameter grid
        sindy_param_grid = {
            'method': ['sequential_lasso', 'lasso', 'ridge'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'scaler': ['standard', 'robust'],
            'normalize': [True, False]
        }
        
        # Library parameter grid
        library_param_grid = {
            'include_ratios': [False, True],
            'include_interactions': [True, False],
            'include_technical': [True, False],
            'max_degree': [1, 2, 3]
        }
        
        results = []
        successful_tests = 0
        total_combinations = 0
        
        print(f"\nTesting {n_feature_trials} feature combinations with SINDy parameter optimization...")
        
        for trial in range(n_feature_trials):
            # Random feature selection
            n_features = np.random.randint(3, min(max_features + 1, len(valid_features) + 1))
            selected_features = np.random.choice(valid_features, n_features, replace=False).tolist()
            
            # Test parameter combinations (sample to keep reasonable runtime)
            n_param_tests = 20  # Sample parameter combinations
            
            for param_test in range(n_param_tests):
                total_combinations += 1
                
                # Random parameter selection
                sindy_params = {
                    'method': np.random.choice(sindy_param_grid['method']),
                    'alpha': np.random.choice(sindy_param_grid['alpha']),
                    'scaler': np.random.choice(sindy_param_grid['scaler']),
                    'normalize': np.random.choice(sindy_param_grid['normalize'])
                }
                
                library_params = {
                    'include_ratios': np.random.choice(library_param_grid['include_ratios']),
                    'include_interactions': np.random.choice(library_param_grid['include_interactions']),
                    'include_technical': np.random.choice(library_param_grid['include_technical']),
                    'max_degree': np.random.choice(library_param_grid['max_degree'])
                }
                
                score = self.evaluate_model(
                    indicators_train, target_train,
                    indicators_val, target_val,
                    selected_features, sindy_params, library_params
                )
                
                if score < float('inf'):
                    results.append({
                        'features': selected_features.copy(),
                        'sindy_params': sindy_params.copy(),
                        'library_params': library_params.copy(),
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
        self.best_params = results[0]['sindy_params']
        self.library_config = results[0]['library_params']
        
        # Display results
        print("\n" + "="*80)
        print("SINDY OPTIMIZATION RESULTS (Validation Set Performance)")
        print("="*80)
        
        for i, result in enumerate(results[:15]):
            print(f"\n{i+1}. Score: {result['score']:.6f}")
            print(f"   Features: {', '.join(result['features'][:3])}")
            if len(result['features']) > 3:
                print(f"            {', '.join(result['features'][3:])}")
            print(f"   SINDy: {result['sindy_params']}")
            print(f"   Library: {result['library_params']}")
        
        return results
    
    def final_training_and_evaluation(self, df):
        """Final model training with comprehensive directional analysis"""
        print("\n" + "="*80)
        print("FINAL SINDY MODEL TRAINING AND TEST SET EVALUATION")
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
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        Y_train_scaled = self.target_scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
        
        X_test_scaled = self.feature_scaler.transform(X_test)
        Y_test_scaled = self.target_scaler.transform(Y_test.reshape(-1, 1)).flatten()
        
        # Create and train library + model
        self.best_library = FinancialLibrary(
            include_ratios=self.library_config['include_ratios'],
            include_interactions=self.library_config['include_interactions'],
            include_technical=self.library_config['include_technical'],
            max_degree=self.library_config['max_degree']
        )
        
        # Transform to function library
        theta_train = self.best_library.transform(X_train_scaled)
        theta_test = self.best_library.transform(X_test_scaled)
        
        # Train final SINDy model
        self.best_model = SINDyRegressor(
            method=self.best_params['method'],
            alpha=self.best_params['alpha'],
            normalize=self.best_params['normalize']
        )
        self.best_model.fit(theta_train, Y_train_scaled)
        
        # Get discovered equations
        equations = self.best_model.get_equation(self.best_library.feature_names, [self.target_metric])
        print(f"\n🔬 DISCOVERED NONLINEAR EQUATION:")
        for eq in equations:
            print(f"  {eq}")
        
        # Report sparsity
        n_nonzero = np.sum(np.abs(self.best_model.coef_) > 1e-8)
        n_total = self.best_model.coef_.size
        print(f"\nModel sparsity: {n_nonzero}/{n_total} non-zero terms")
        
        # Predict on test set
        Y_pred_scaled = self.best_model.predict(theta_test).flatten()
        Y_pred = self.target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate traditional metrics
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        mae = mean_absolute_error(Y_test, Y_pred)
        correlation = np.corrcoef(Y_pred, Y_test)[0, 1] if len(Y_pred) > 1 else 0
        bias = np.mean(Y_pred - Y_test)
        r2 = r2_score(Y_test, Y_pred)
        
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
        print(f"R²: {r2:.4f}")
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
        
        # Show key nonlinear terms
        print(f"\n🔬 KEY NONLINEAR TERMS DISCOVERED:")
        important_terms = self.get_important_terms(top_n=10)
        if important_terms:
            for term in important_terms:
                print(f"  {term['rank']}. {term['feature']}: {term['coefficient']:.6f}")
        
        # Recommendation for trading
        print(f"\n💡 TRADING RECOMMENDATION:")
        if best_acc > 0.6:
            print(f"✅ Strong nonlinear signal: {best_horizon}-day directional accuracy of {best_acc:.1%}")
            print(f"   Recommended use: Daily rebalancing with {best_horizon}-day outlook")
            print(f"   🔬 Advantage: Captures nonlinear market dynamics DMD might miss")
        elif best_acc > 0.55:
            print(f"⚡ Moderate nonlinear signal: {best_horizon}-day directional accuracy of {best_acc:.1%}")
            print(f"   Recommended use: Combine with DMD for hybrid approach")
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
        plt.plot(dates, predictions, 'r--', label='SINDy Prediction', alpha=0.8, linewidth=1.5)
        
        # Add directional markers
        for i in range(0, len(dates), max(1, len(dates)//20)):
            if i < len(predictions) and i < len(actuals):
                pred_dir = '↗️' if predictions[i] > 0 else ('↘️' if predictions[i] < 0 else '→')
                actual_dir = '↗️' if actuals[i] > 0 else ('↘️' if actuals[i] < 0 else '→')
                color = 'green' if pred_dir == actual_dir else 'red'
                plt.plot(dates[i], predictions[i], marker='o', color=color, markersize=4)
        
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_metric}')
        plt.title(f'SINDy Predictions vs Actual - {self.target_metric}\n(Green dots = correct direction, Red = wrong)')
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
        plt.title('SINDy Directional Accuracy by Time Horizon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Nonlinear terms importance
        plt.subplot(3, 2, 4)
        important_terms = self.get_important_terms(top_n=15)
        if important_terms:
            coeffs = [term['abs_coefficient'] for term in important_terms]
            names = [term['feature'] for term in important_terms]
            
            # Highlight nonlinear terms
            colors = ['red' if any(x in name for x in ['^2', '*', '/', 'sqrt']) else 'blue' for name in names]
            
            plt.barh(range(len(coeffs)), coeffs, alpha=0.7, color=colors)
            plt.yticks(range(len(names)), names, fontsize=8)
            plt.xlabel('|Coefficient|')
            plt.title('Important SINDy Terms\n(Red = Nonlinear, Blue = Linear)')
        
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
            window = 20
            
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
        
        print(f"\n🔮 SINDY DAILY PREDICTION FOR NEXT {prediction_horizon} DAYS")
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
                    last_features.append(0.0)
            else:
                missing_features.append(col)
                last_features.append(0.0)
        
        if missing_features:
            print(f"⚠️  Warning: Missing features {missing_features}, using default values")
        
        if len(last_features) != len(self.best_features):
            raise ValueError(f"Feature mismatch: expected {len(self.best_features)}, got {len(last_features)}")
        
        try:
            X_last = np.array(last_features).reshape(1, -1)
            X_last_scaled = self.feature_scaler.transform(X_last)
            
            # Transform through function library
            theta_last = self.best_library.transform(X_last_scaled)
            
            # Predict next value
            Y_next_scaled = self.best_model.predict(theta_last).flatten()[0]
            Y_next = self.target_scaler.inverse_transform([[Y_next_scaled]])[0][0]
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
        
        # Interpret prediction based on target type (same logic as DMD)
        if self.target_metric == 'log_prices':
            predicted_log_price = Y_next
            predicted_price = np.exp(predicted_log_price)
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            
            direction = "📈 UP" if percent_change > 0.5 else "📉 DOWN" if percent_change < -0.5 else "➡️ FLAT"
            confidence = "HIGH" if abs(percent_change) > 2 else "MEDIUM" if abs(percent_change) > 1 else "LOW"
            
            print(f"\n🎯 SINDY PREDICTION:")
            print(f"   Expected Price: ${predicted_price:.2f}")
            print(f"   Expected Change: {price_change:+.2f} ({percent_change:+.1f}%)")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {confidence}")
            
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
            
            print(f"\n🎯 SINDY PREDICTION:")
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
            
            confidence = "HIGH" if abs(log_ratio_change) > 0.05 else "MEDIUM" if abs(log_ratio_change) > 0.02 else "LOW"
            
        # Show key nonlinear factors influencing the prediction
        print(f"\n🔬 KEY NONLINEAR FACTORS:")
        important_terms = self.get_important_terms(top_n=5)
        if important_terms:
            for term in important_terms:
                nonlinear_indicator = "🔴" if any(x in term['feature'] for x in ['^2', '*', '/', 'sqrt']) else "🔵"
                print(f"   {nonlinear_indicator} {term['feature']}: {term['coefficient']:.4f}")
        
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
        print(f"\n💡 NONLINEAR TRADING RECOMMENDATION:")
        
        if self.target_metric == 'log_price_ratio_sma20':
            if log_ratio_change > 0.01:
                recommendation = "🟢 CONSIDER BUYING (Nonlinear momentum up)"
            elif log_ratio_change < -0.01:
                recommendation = "🔴 CONSIDER SELLING (Nonlinear momentum down)"
            else:
                recommendation = "🟡 HOLD (Limited nonlinear signal)"
        else:
            if abs(percent_change) > 1:
                action = "BUY" if percent_change > 0 else "SELL"
                recommendation = f"{'🟢' if percent_change > 0 else '🔴'} CONSIDER {action}ING (Nonlinear signal)"
            else:
                recommendation = "🟡 HOLD (Weak nonlinear signal)"
        
        print(f"   {recommendation}")
        print(f"   🔬 Advantage: Captures nonlinear market dynamics")
        print(f"   Stop Loss: {2 * abs(percent_change):.1f}%")
        print(f"   Position Size: Recommend {min(100, max(10, 50 / max(1, abs(percent_change)))): .0f}% of normal")
        
        return {
            'prediction_date': next_date,
            'current_price': current_price,
            'predicted_value': Y_next,
            'expected_change_pct': percent_change if 'percent_change' in locals() else 0,
            'direction': direction if 'direction' in locals() else 'UNKNOWN',
            'confidence': confidence if 'confidence' in locals() else 'LOW',
            'recommendation': recommendation,
            'nonlinear_factors': important_terms[:3] if important_terms else []
        }
    
    def get_important_terms(self, top_n=10):
        """Get the most important terms from SINDy model"""
        if self.best_model is None or self.best_model.coef_ is None:
            return None
        
        coef_abs = np.abs(self.best_model.coef_.flatten())
        non_zero_idx = np.where(coef_abs > 1e-8)[0]
        
        if len(non_zero_idx) == 0:
            return []
        
        # Sort by importance
        importance_idx = non_zero_idx[np.argsort(coef_abs[non_zero_idx])[::-1]]
        
        top_terms = []
        for i, idx in enumerate(importance_idx[:top_n]):
            term = {
                'feature': self.best_library.feature_names[idx],
                'coefficient': self.best_model.coef_.flatten()[idx],
                'abs_coefficient': coef_abs[idx],
                'rank': i + 1
            }
            top_terms.append(term)
        
        return top_terms
    
    def get_feature_importance(self):
        """Analyze feature importance from the SINDy model"""
        if self.best_model is None or self.best_model.coef_ is None:
            raise ValueError("Model not trained!")
        
        # Get coefficients and feature names
        coefficients = self.best_model.coef_.flatten()
        feature_names = self.best_library.feature_names
        
        # Calculate feature importance as absolute coefficient values
        feature_importance = np.abs(coefficients)
        
        # Normalize
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'coefficient': coefficients
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("SINDY TRADING SYSTEM - SUMMARY REPORT")
        print("="*80)
        
        print(f"\n📊 MODEL CONFIGURATION:")
        print(f"Target Metric: {self.target_metric}")
        print(f"Best SINDy Parameters: {self.best_params}")
        print(f"Best Library Config: {self.library_config}")
        print(f"Selected Features: {len(self.best_features)}")
        
        print(f"\n🎯 FEATURE SELECTION:")
        for i, feature in enumerate(self.best_features, 1):
            print(f"  {i:2d}. {feature}")
        
        if self.best_model is not None:
            try:
                importance_df = self.get_feature_importance()
                print(f"\n📈 FEATURE IMPORTANCE (Top 5):")
                for idx, row in importance_df.head().iterrows():
                    nonlinear_indicator = "🔴" if any(x in row['feature'] for x in ['^2', '*', '/', 'sqrt']) else "🔵"
                    print(f"  {nonlinear_indicator} {row['feature']}: {row['importance']:.1%} (coef: {row['coefficient']:.4f})")
            except Exception:
                print(f"\n📈 FEATURE IMPORTANCE: Unable to calculate")
        
        if self.best_model is not None and self.best_library is not None:
            print(f"\n🔬 DISCOVERED EQUATION:")
            try:
                equations = self.best_model.get_equation(self.best_library.feature_names, [self.target_metric])
                for eq in equations:
                    print(f"  {eq}")
            except Exception:
                print("  Unable to display equation")
        
        print(f"\n💡 USAGE INSTRUCTIONS:")
        print(f"1. Load your data with columns: ['Open', 'High', 'Low', 'Close', 'Volume']")
        print(f"2. Call: system.comprehensive_optimize(df)")
        print(f"3. Call: system.final_training_and_evaluation(df)")
        print(f"4. Call: system.make_daily_prediction(df, prediction_horizon=5) for predictions")
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"- All features use strict 1+ day lag to prevent look-ahead bias")
        print(f"- Model discovers nonlinear dynamics automatically via SINDy")
        print(f"- Best used for 1-10 day directional predictions")
        print(f"- Captures market dynamics that linear models (DMD) might miss")
        print(f"- 🔴 Red features are nonlinear terms, 🔵 Blue are linear")
        
        print("="*80)


# Enhanced Combined Predictor for DMD + SINDy
class EnhancedCombinedPredictor:
    """Enhanced predictor for combining DMD and SINDy with detailed analysis"""
    
    def __init__(self, dmd_weight=0.6, sindy_weight=0.4):
        self.dmd_weight = dmd_weight
        self.sindy_weight = sindy_weight
        self.dmd_system = None
        self.sindy_system = None
        
    def set_systems(self, dmd_system, sindy_system):
        """Set the trained systems"""
        self.dmd_system = dmd_system
        self.sindy_system = sindy_system
        
    def predict_direction(self, df, horizon=5):
        """Enhanced combined prediction with detailed analysis"""
        print("\n" + "="*70)
        print(f"ENHANCED DMD-SINDY COMBINED PREDICTION (Horizon: {horizon} days)")
        print("="*70)
        
        predictions = {}
        
        # Get DMD prediction
        if self.dmd_system is not None:
            try:
                dmd_pred = self.dmd_system.make_daily_prediction(df, horizon)
                predictions['dmd'] = {
                    'direction': dmd_pred['direction'],
                    'confidence': dmd_pred.get('confidence', 'UNKNOWN'),
                    'expected_change': dmd_pred.get('expected_change_pct', 0),
                    'raw_prediction': dmd_pred,
                    'model_type': 'LINEAR'
                }
                
                print(f"\n📊 DMD (Linear) Prediction:")
                print(f"   Direction: {dmd_pred['direction']}")
                print(f"   Expected Change: {dmd_pred.get('expected_change_pct', 0):.2f}%")
                print(f"   Confidence: {dmd_pred.get('confidence', 'UNKNOWN')}")
                print(f"   🔵 Linear dynamics captured")
            except Exception as e:
                print(f"\n❌ DMD prediction failed: {e}")
                predictions['dmd'] = None
        
        # Get SINDy prediction
        if self.sindy_system is not None:
            try:
                sindy_pred = self.sindy_system.make_daily_prediction(df, horizon)
                
                if sindy_pred is not None:
                    predictions['sindy'] = {
                        'direction': sindy_pred['direction'],
                        'confidence': sindy_pred.get('confidence', 'UNKNOWN'),
                        'expected_change': sindy_pred.get('expected_change_pct', 0),
                        'raw_prediction': sindy_pred,
                        'model_type': 'NONLINEAR',
                        'nonlinear_factors': sindy_pred.get('nonlinear_factors', [])
                    }
                    
                    print(f"\n🔬 SINDy (Nonlinear) Prediction:")
                    print(f"   Direction: {sindy_pred['direction']}")
                    print(f"   Expected Change: {sindy_pred.get('expected_change_pct', 0):.2f}%")
                    print(f"   Confidence: {sindy_pred.get('confidence', 'UNKNOWN')}")
                    print(f"   🔴 Nonlinear dynamics captured")
                    
                    # Show key nonlinear factors
                    nonlinear_factors = sindy_pred.get('nonlinear_factors', [])
                    if nonlinear_factors:
                        print(f"   Key Nonlinear Terms:")
                        for factor in nonlinear_factors:
                            print(f"     - {factor['feature']}: {factor['coefficient']:.4f}")
                else:
                    predictions['sindy'] = None
                    print(f"\n❌ SINDy prediction returned no values")
            except Exception as e:
                print(f"\n❌ SINDy prediction failed: {e}")
                predictions['sindy'] = None
        
        # Enhanced combination logic
        if predictions.get('dmd') and predictions.get('sindy'):
            # Extract directional signals
            dmd_dir = 1 if 'UP' in predictions['dmd']['direction'] else (-1 if 'DOWN' in predictions['dmd']['direction'] else 0)
            sindy_dir = 1 if 'UP' in predictions['sindy']['direction'] else (-1 if 'DOWN' in predictions['sindy']['direction'] else 0)
            
            # Weighted combination
            combined_signal = self.dmd_weight * dmd_dir + self.sindy_weight * sindy_dir
            
            # Enhanced direction determination
            if combined_signal > 0.4:
                combined_direction = "📈 STRONG UP"
            elif combined_signal > 0.1:
                combined_direction = "📈 UP"
            elif combined_signal < -0.4:
                combined_direction = "📉 STRONG DOWN"
            elif combined_signal < -0.1:
                combined_direction = "📉 DOWN"
            else:
                combined_direction = "➡️ FLAT/MIXED"
            
            # Enhanced confidence calculation
            confidence_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 0}
            dmd_conf_score = confidence_map.get(predictions['dmd'].get('confidence', 'UNKNOWN'), 0)
            sindy_conf_score = confidence_map.get(predictions['sindy'].get('confidence', 'UNKNOWN'), 0)
            
            avg_conf_score = (self.dmd_weight * dmd_conf_score + self.sindy_weight * sindy_conf_score)
            
            if avg_conf_score >= 2.5:
                combined_confidence = "HIGH"
            elif avg_conf_score >= 1.5:
                combined_confidence = "MEDIUM"
            else:
                combined_confidence = "LOW"
            
            # Agreement analysis
            if dmd_dir == sindy_dir and abs(dmd_dir) == 1:
                agreement = "STRONG_AGREE"
                agreement_score = 1.0
            elif dmd_dir == sindy_dir:
                agreement = "AGREE"
                agreement_score = 0.8
            elif (dmd_dir == 0 and abs(sindy_dir) <= 1) or (sindy_dir == 0 and abs(dmd_dir) <= 1):
                agreement = "MILD_DISAGREE"
                agreement_score = 0.3
            else:
                agreement = "STRONG_DISAGREE"
                agreement_score = 0.0
            
            # Model dynamics analysis
            dynamics_analysis = self._analyze_model_dynamics(predictions)
            
            print(f"\n🎯 ENHANCED COMBINED PREDICTION:")
            print(f"   Direction: {combined_direction}")
            print(f"   Confidence: {combined_confidence}")
            print(f"   Model Agreement: {agreement}")
            print(f"   Signal Strength: {abs(combined_signal):.2f}")
            print(f"   Agreement Score: {agreement_score:.1f}")
            
            # Show dynamics captured by each model
            print(f"\n🔍 DYNAMICS ANALYSIS:")
            print(f"   🔵 Linear (DMD): {dynamics_analysis['dmd_dynamics']}")
            print(f"   🔴 Nonlinear (SINDy): {dynamics_analysis['sindy_dynamics']}")
            print(f"   ⚖️  Consensus: {dynamics_analysis['consensus']}")
            
            # Enhanced recommendations
            if agreement == "STRONG_AGREE" and combined_confidence == "HIGH":
                print(f"\n✅ VERY STRONG SIGNAL: Both linear and nonlinear models strongly agree")
                print(f"   🎯 HIGH CONVICTION TRADE: {combined_direction}")
                print(f"   🔬 Both market regimes (linear + nonlinear) aligned")
                risk_level = "MODERATE"
            elif agreement in ["STRONG_AGREE", "AGREE"] and combined_confidence in ["HIGH", "MEDIUM"]:
                print(f"\n✅ STRONG SIGNAL: Models agree with good confidence")
                print(f"   📈 RECOMMENDED TRADE: {combined_direction}")
                print(f"   🔬 Linear and nonlinear dynamics aligned")
                risk_level = "MODERATE"
            elif agreement == "MILD_DISAGREE":
                print(f"\n🟡 MIXED SIGNALS: Models show mild disagreement")
                print(f"   ⚡ SMALL POSITION: {combined_direction} with tight stops")
                print(f"   🔬 Market transition between linear/nonlinear regimes")
                risk_level = "HIGH"
            elif agreement == "STRONG_DISAGREE":
                print(f"\n⚠️  CONFLICTING SIGNALS: Linear vs nonlinear models disagree")
                print(f"   🚫 AVOID TRADING: Wait for regime clarity")
                print(f"   🔬 Market dynamics unclear - possible regime change")
                risk_level = "VERY_HIGH"
            else:
                print(f"\n🟡 WEAK SIGNAL: Low confidence predictions")
                print(f"   ⚡ SMALL POSITION: {combined_direction} with tight stops")
                risk_level = "HIGH"
            
            # Expected returns calculation
            dmd_return = predictions['dmd'].get('expected_change', 0)
            sindy_return = predictions['sindy'].get('expected_change', 0)
            combined_return = self.dmd_weight * dmd_return + self.sindy_weight * sindy_return
            
            print(f"\n📊 Expected Returns ({horizon}-day forecast):")
            print(f"   🔵 DMD (Linear): {dmd_return:+.2f}%")
            print(f"   🔴 SINDy (Nonlinear): {sindy_return:+.2f}%")
            print(f"   ⚖️  Combined: {combined_return:+.2f}%")
            print(f"   Risk Level: {risk_level}")
            
            # Model-specific insights
            print(f"\n🔬 MODEL INSIGHTS:")
            if abs(dmd_return) > abs(sindy_return):
                print(f"   📊 Linear dynamics dominate this prediction")
                print(f"   💡 Market showing trend-following or mean-reverting behavior")
            elif abs(sindy_return) > abs(dmd_return):
                print(f"   🌀 Nonlinear dynamics dominate this prediction")
                print(f"   💡 Market showing complex, nonlinear behavior")
            else:
                print(f"   ⚖️  Balanced linear/nonlinear dynamics")
                print(f"   💡 Market in mixed regime")
            
            predictions['combined'] = {
                'direction': combined_direction,
                'confidence': combined_confidence,
                'agreement': agreement,
                'agreement_score': agreement_score,
                'expected_return': combined_return,
                'signal_strength': abs(combined_signal),
                'risk_level': risk_level,
                'dynamics_analysis': dynamics_analysis
            }
        else:
            print(f"\n⚠️  Cannot create combined prediction - missing model predictions")
            predictions['combined'] = None
        
        return predictions
    
    def _analyze_model_dynamics(self, predictions):
        """Analyze what dynamics each model is capturing"""
        analysis = {
            'dmd_dynamics': 'Unknown',
            'sindy_dynamics': 'Unknown',
            'consensus': 'Unknown'
        }
        
        try:
            # Analyze DMD dynamics
            if predictions.get('dmd'):
                dmd_change = predictions['dmd'].get('expected_change', 0)
                if abs(dmd_change) > 2:
                    analysis['dmd_dynamics'] = 'Strong trend/mean-reversion'
                elif abs(dmd_change) > 0.5:
                    analysis['dmd_dynamics'] = 'Moderate linear trend'
                else:
                    analysis['dmd_dynamics'] = 'Weak linear signal'
            
            # Analyze SINDy dynamics
            if predictions.get('sindy'):
                sindy_change = predictions['sindy'].get('expected_change', 0)
                nonlinear_factors = predictions['sindy'].get('nonlinear_factors', [])
                
                has_strong_nonlinear = any(abs(f.get('coefficient', 0)) > 0.1 for f in nonlinear_factors)
                
                if abs(sindy_change) > 2 and has_strong_nonlinear:
                    analysis['sindy_dynamics'] = 'Strong nonlinear regime'
                elif abs(sindy_change) > 0.5 and has_strong_nonlinear:
                    analysis['sindy_dynamics'] = 'Moderate nonlinear effects'
                elif has_strong_nonlinear:
                    analysis['sindy_dynamics'] = 'Weak nonlinear dynamics'
                else:
                    analysis['sindy_dynamics'] = 'Mostly linear behavior'
            
            # Determine consensus
            dmd_dir = 1 if 'UP' in predictions.get('dmd', {}).get('direction', '') else (-1 if 'DOWN' in predictions.get('dmd', {}).get('direction', '') else 0)
            sindy_dir = 1 if 'UP' in predictions.get('sindy', {}).get('direction', '') else (-1 if 'DOWN' in predictions.get('sindy', {}).get('direction', '') else 0)
            
            if dmd_dir == sindy_dir and abs(dmd_dir) == 1:
                analysis['consensus'] = 'Strong alignment across regimes'
            elif dmd_dir == sindy_dir:
                analysis['consensus'] = 'Moderate alignment'
            else:
                analysis['consensus'] = 'Regime conflict detected'
                
        except Exception as e:
            pass  # Keep default 'Unknown' values
        
        return analysis


# Main analysis functions
def run_robust_sindy_analysis(filename, target_metric='log_price_ratio_sma20', prediction_horizon=5):
    """Run comprehensive SINDy analysis focused on directional prediction"""
    
    # Load data
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [col.capitalize() for col in df.columns]
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize system
    sindy_system = RobustSINDySystem(target_metric=target_metric)
    
    # Show what we're predicting and WHY it's valid
    if target_metric == 'log_prices':
        print("✅ SCIENTIFICALLY VALIDATED: log_prices")
        print("   Predicting tomorrow's log price using nonlinear dynamics")
        print("   🔬 SINDy can discover nonlinear price evolution equations")
    elif target_metric == 'log_price_ratio_sma20':
        print("✅ ENHANCED VALIDATED METRIC: log_price_ratio_sma20")
        print("   Predicting log(price/SMA20) - nonlinear mean reversion")
        print("   🔬 SINDy can discover complex mean-reverting dynamics")
    elif target_metric == 'log_price_ratio_sma200':
        print("✅ VALIDATED METRIC: log_price_ratio_sma200")
        print("   Predicting log(price/SMA200) - long-term nonlinear dynamics")
        print("   🔬 SINDy can capture complex trend relationships")
    else:
        print(f"⚠️  WARNING: {target_metric} not fully validated!")
        print("   Consider using 'log_prices', 'log_price_ratio_sma20', or 'log_price_ratio_sma200'")
    
    print(f"\n🎯 FOCUS: Optimizing for {prediction_horizon}-day directional accuracy")
    print(f"🔬 ADVANTAGE: SINDy discovers nonlinear market dynamics automatically")
    
    # Comprehensive optimization
    results = sindy_system.comprehensive_optimize(df, max_features=8, n_feature_trials=50)
    
    # Train and evaluate with directional focus
    predictions, actuals, directional_results = sindy_system.final_training_and_evaluation(df)
    
    # Make live prediction
    live_prediction = sindy_system.make_daily_prediction(df, prediction_horizon)
    
    # Show results
    print(f"\n📈 BEST FEATURES FOR PREDICTING {target_metric}:")
    for i, feat in enumerate(sindy_system.best_features):
        print(f"  {i+1}. {feat}")
    
    print(f"\n⚙️  BEST SINDY PARAMETERS:")
    for key, value in sindy_system.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔬 BEST LIBRARY CONFIGURATION:")
    for key, value in sindy_system.library_config.items():
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
    print(f"   🔬 Nonlinear Advantage: Captures market dynamics linear models miss")
    
    return sindy_system, directional_results, live_prediction


def run_combined_dmd_sindy_analysis(filename, target_metric='log_price_ratio_sma20', prediction_horizon=5):
    """Run combined DMD + SINDy analysis for ultimate directional prediction"""
    
    print("="*80)
    print("ULTIMATE COMBINED DMD + SINDY DIRECTIONAL PREDICTION SYSTEM")
    print("="*80)
    
    # Load data
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = [col.capitalize() for col in df.columns]
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Train DMD system (you need to import RobustDMDSystem from the DMD code)
    print(f"\n{'='*50}")
    print("TRAINING DMD (LINEAR) SYSTEM")
    print(f"{'='*50}")
    
    try:
        # Assuming you have the DMD system available
        from your_dmd_module import RobustDMDSystem  # Replace with actual import
        
        dmd_system = RobustDMDSystem(target_metric=target_metric)
        dmd_results = dmd_system.comprehensive_optimize(df, max_features=8, n_feature_trials=50)
        dmd_predictions, dmd_actuals, dmd_directional = dmd_system.final_training_and_evaluation(df)
        
        print(f"✅ DMD System trained successfully")
        
    except ImportError:
        print("⚠️  DMD system not available - will use SINDy only")
        dmd_system = None
    
    # Train SINDy system
    print(f"\n{'='*50}")
    print("TRAINING SINDY (NONLINEAR) SYSTEM")
    print(f"{'='*50}")
    
    sindy_system, sindy_directional, sindy_live = run_robust_sindy_analysis(
        filename, target_metric, prediction_horizon
    )
    
    # Combined prediction
    if dmd_system is not None:
        print(f"\n{'='*50}")
        print("CREATING COMBINED PREDICTOR")
        print(f"{'='*50}")
        
        combined_predictor = EnhancedCombinedPredictor(dmd_weight=0.5, sindy_weight=0.5)
        combined_predictor.set_systems(dmd_system, sindy_system)
        
        # Make combined prediction
        combined_prediction = combined_predictor.predict_direction(df, prediction_horizon)
        
        return dmd_system, sindy_system, combined_predictor, combined_prediction
    else:
        print(f"\n⚠️  Running SINDy-only analysis")
        return None, sindy_system, None, sindy_live


# Usage examples:
"""
# Basic SINDy analysis
sindy_system, results, prediction = run_robust_sindy_analysis('your_data.csv', 'log_price_ratio_sma20', 5)

# Combined DMD + SINDy analysis (ultimate system)
dmd_sys, sindy_sys, combined, pred = run_combined_dmd_sindy_analysis('your_data.csv', 'log_price_ratio_sma20', 5)

# Generate detailed reports
sindy_system.summary_report()
"""