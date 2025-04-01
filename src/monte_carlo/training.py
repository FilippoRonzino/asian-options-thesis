import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from mcpricer import MCPricer
from asianoption import AsianOption

class LSTMOptionPricer:
    """
    Neural network pricing model for options using LSTM architecture.
    """
    
    def __init__(self, lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001, 
                 bidirectional=False, lookback_window=20, option_type='regular'):
        """
        Initialize the LSTM option pricer.
        
        Parameters:
        -----------
        lstm_units : list
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        bidirectional : bool
            Whether to use bidirectional LSTM layers
        lookback_window : int
            Number of time steps to look back for time series data
        option_type : str
            Type of option to price ('regular', 'asian', etc.)
        """
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.lookback_window = lookback_window
        self.option_type = option_type
        self.model = None
        self.feature_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.is_trained = False
    
    def _build_model(self, input_shape):
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        if self.bidirectional:
            model.add(Bidirectional(
                LSTM(self.lstm_units[0], return_sequences=len(self.lstm_units) > 1),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                self.lstm_units[0], 
                return_sequences=len(self.lstm_units) > 1,
                input_shape=input_shape
            ))
        
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units) - 2
            
            if self.bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _prepare_time_series(self, option_configs, prices=None):
        """Convert option configurations to time series format for LSTM."""
        # Generate features from option configurations
        features = []
        for config in option_configs:
            # Extract basic option parameters
            S0 = config['S0']
            K = config['K']
            T = config['T']
            r = config['r']
            sigma = config['sigma']
            
            # Common derived features
            moneyness = S0 / K
            log_moneyness = np.log(moneyness)
            sigma_sqrt_t = sigma * np.sqrt(T)
            time_decay_factor = np.exp(-r * T)
            
            # Base feature vector
            feature_vector = [S0, K, T, r, sigma, moneyness, log_moneyness, sigma_sqrt_t, time_decay_factor]
            
            # Add Asian-specific features if needed
            # if self.option_type == 'asian':
            #     adjusted_sigma = sigma / np.sqrt(3)  # Common approximation for arithmetic Asian
            #     adjusted_sigma_sqrt_t = adjusted_sigma * np.sqrt(T)
            #     feature_vector.extend([adjusted_sigma, adjusted_sigma_sqrt_t])
                
            features.append(feature_vector)
        
        # Scale features
        features = np.array(features)
        if not self.is_trained:
            self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        # Create time series data
        X = []
        
        for i, feature in enumerate(features_scaled):
            # Create a time series for this option
            time_series = []
            config = option_configs[i]
            
            # Generate synthetic time points for this option
            for j in range(self.lookback_window):
                # Time fraction (0 to 1)
                time_frac = j / (self.lookback_window - 1)
                
                # Adjust time to maturity
                adjusted_T = config['T'] * (1 - time_frac)
                
                # Recalculate features with the adjusted time
                adjusted_feature = self._create_adjusted_feature(config, adjusted_T, time_frac)
                
                # Scale this adjusted feature
                adjusted_feature_scaled = self.feature_scaler.transform([adjusted_feature])[0]
                time_series.append(adjusted_feature_scaled)
            
            X.append(time_series)
        
        X = np.array(X)
        
        if prices is not None:
            # If prices are provided, prepare target values
            prices = np.array(prices).reshape(-1, 1)
            
            if not self.is_trained:
                self.price_scaler.fit(prices)
            
            y = self.price_scaler.transform(prices)
            return X, y
        else:
            return X
    
    def _create_adjusted_feature(self, config, adjusted_T, time_frac):
        """Create adjusted feature vector with new time to maturity."""
        S0 = config['S0']
        K = config['K']
        r = config['r']
        sigma = config['sigma']
        
        # Common features
        moneyness = S0 / K
        log_moneyness = np.log(moneyness)
        sigma_sqrt_t = sigma * np.sqrt(adjusted_T)
        adjusted_time_decay = np.exp(-r * adjusted_T)
        
        # Base feature vector
        adjusted_feature = [S0, K, adjusted_T, r, sigma, moneyness, log_moneyness, sigma_sqrt_t, adjusted_time_decay]
        
        # Add Asian-specific features if needed
        # if self.option_type == 'asian':
        #     adjusted_sigma = sigma / np.sqrt(3)
        #     adjusted_sigma_sqrt_t = adjusted_sigma * np.sqrt(adjusted_T)
        #     adjusted_feature.extend([adjusted_sigma, adjusted_sigma_sqrt_t])
            
        return adjusted_feature
    
    def train(self, option_configs, prices, validation_split=0.4, epochs=100, batch_size=32, verbose=1):
        """Train the LSTM model on option prices."""
        # Prepare time series data
        X, y = self._prepare_time_series(option_configs, prices)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Build the model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self._build_model(input_shape)
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]
        
        # Train the model
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        training_time = time.time() - start_time
        
        self.is_trained = True
        print(f"LSTM model trained in {training_time:.2f} seconds")
        
        return {
            'history': history.history,
            'training_time': training_time
        }
    
    def predict_price(self, option_configs):
        """Predict option prices using the trained LSTM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare time series data
        X = self._prepare_time_series(option_configs)
        
        # Predict
        start_time = time.time()
        y_pred_scaled = self.model.predict(X)
        prediction_time = time.time() - start_time
        
        # Inverse transform to get actual prices
        y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten(), prediction_time


def custom_prepare_option_data(df):
    """Custom function to prepare option configurations from market data DataFrame."""
    option_configs = []
    market_prices = []
    
    # Filter for call options and valid data
    filtered_df = df[
        (df['TIME_TO_MATURITY'] > 0) &
        (df['OPT_PX'] > 0) &
        (df['STRIKE_PX'] > 0) &
        (df['UNDERLYING_PRICE'] > 0)
    ]
    
    for idx, row in filtered_df.iterrows():
        # Get volatility
        sigma = row['HIST_CALL_IMP_VOL']
        
        # Fallback to alternative volatility if needed
        if pd.isna(sigma) or sigma <= 0:
            # Try using alternative volatility metrics
            for vol_column in ['VOLATILITY_30D', 'VOLATILITY_20D', 'VOLATILITY_60D', 'VOLATILITY_90D', 'VOLATILITY_10D']:
                if vol_column in row and not pd.isna(row[vol_column]) and row[vol_column] > 0:
                    sigma = row[vol_column]
                    break
            
            # Skip if no valid volatility found
            if pd.isna(sigma) or sigma <= 0:
                continue
        
        # Convert volatility from percentage if needed
        if sigma > 1:
            sigma = sigma / 100
        
        # Create option configuration
        config = {
            'name': row.get('SECURITY_DES', f'Option_{idx}'),
            'S0': row['UNDERLYING_PRICE'],
            'K': row['STRIKE_PX'],
            'T': row['TIME_TO_MATURITY'],
            'r': row['RISK_FREE_RATE'],
            'sigma': sigma,
            'option_type': 'arithmetic'  # For Asian options
        }
        
        option_configs.append(config)
        market_prices.append(row['OPT_PX'])
    
    return option_configs, market_prices


# Modify the evaluate_lstm_model function to return results without plotting
def evaluate_lstm_model(df, lstm_configs=None, mc_pricer=None, asian_option_class=None, 
                        option_type='asian', plot=False, cv = True): 
    """
    Evaluate LSTM models against market data and Monte Carlo pricing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing option market data
    lstm_configs : list of dict
        List of LSTM configurations to compare
    mc_pricer : MCPricer
        Monte Carlo pricer instance
    asian_option_class : class
        Asian option class
    option_type : str
        Type of option to price
    plot : bool
        Whether to plot results immediately (default: False)
    """
    if lstm_configs is None:
        lstm_configs = [
            {
                'name': 'Basic LSTM',
                'params': {
                    'lstm_units': [64, 32],
                    'dropout_rate': 0.2,
                    'bidirectional': False,
                    'option_type': option_type
                }
            },
            {
                'name': 'Deep LSTM',
                'params': {
                    'lstm_units': [128, 64, 32],
                    'dropout_rate': 0.3,
                    'bidirectional': False,
                    'option_type': option_type
                }
            },
            {
                'name': 'Bidirectional LSTM',
                'params': {
                    'lstm_units': [64, 32],
                    'dropout_rate': 0.2,
                    'bidirectional': True,
                    'option_type': option_type
                }
            }
        ]
    
    # Prepare option data using our custom function
    option_configs, market_prices = custom_prepare_option_data(df)
    
    if len(option_configs) == 0:
        raise ValueError(f"No valid {option_type} option data found after filtering")
    
    print(f"Prepared {len(option_configs)} valid option configurations from market data")
    
    # Split data
    train_idx, test_idx = train_test_split(range(len(option_configs)), test_size=0.4, random_state=42)
    
    train_configs = [option_configs[i] for i in train_idx]
    train_prices = [market_prices[i] for i in train_idx]
    
    test_configs = [option_configs[i] for i in test_idx]
    test_prices = [market_prices[i] for i in test_idx]
    
    # Calculate Monte Carlo prices if needed
    mc_prices = []
    mc_times = []
    
    if mc_pricer is not None and asian_option_class is not None:
        print("Generating Monte Carlo prices for comparison...")
        for i, config in enumerate(test_configs):
            print(f"Processing option {i+1}/{len(test_configs)}")
            
            # Create option and price it
            option = asian_option_class(
                config['S0'], config['K'], config['T'], 
                config['r'], config['sigma'], 
                n_steps=252, option_type=config['option_type']
            )
            
            start_time = time.time()
            price = mc_pricer.price(option, use_control_variate = cv)
            mc_time = time.time() - start_time
            
            mc_prices.append(price)
            mc_times.append(mc_time)
    
    # Train and evaluate each LSTM model
    lstm_results = {}
    
    for config in lstm_configs:
        name = config['name']
        params = config['params']
        
        print(f"\nTraining {name}...")
        
        # Create and train the model
        lstm_pricer = LSTMOptionPricer(**params)
        training_result = lstm_pricer.train(train_configs, train_prices, epochs=100, verbose=1)
        
        # Predict prices
        lstm_prices, prediction_time = lstm_pricer.predict_price(test_configs)
        
        # Calculate metrics vs market prices
        market_abs_errors = np.abs(lstm_prices - test_prices)
        market_rel_errors = market_abs_errors / np.array(test_prices)
        
        results = {
            'prices': lstm_prices,
            'market_prices': test_prices,
            'market_abs_errors': market_abs_errors,
            'market_rel_errors': market_rel_errors,
            'mean_market_abs_error': np.mean(market_abs_errors),
            'mean_market_rel_error': np.mean(market_rel_errors),
            'market_rmse': np.sqrt(np.mean(np.square(market_abs_errors))),
            'prediction_time': prediction_time,
            'prediction_time_per_option': prediction_time / len(test_configs),
            'training_result': training_result
        }
        
        # Add Monte Carlo comparison if available
        if mc_prices:
            mc_abs_errors = np.abs(lstm_prices - mc_prices)
            mc_rel_errors = mc_abs_errors / np.array(mc_prices)
            
            results.update({
                'mc_prices': mc_prices,
                'mc_abs_errors': mc_abs_errors,
                'mc_rel_errors': mc_rel_errors,
                'mean_mc_abs_error': np.mean(mc_abs_errors),
                'mean_mc_rel_error': np.mean(mc_rel_errors),
                'mc_rmse': np.sqrt(np.mean(np.square(mc_abs_errors))),
                'speedup_vs_mc': np.mean(mc_times) / (prediction_time / len(test_configs))
            })
        
        lstm_results[name] = results
        
        # Print results summary
        print(f"{name} test results:")
        print(f"  Against market prices:")
        print(f"    Mean absolute error: {results['mean_market_abs_error']:.6f}")
        print(f"    Mean relative error: {results['mean_market_rel_error']:.6f}")
        print(f"    RMSE: {results['market_rmse']:.6f}")
        
        if mc_prices:
            print(f"  Against Monte Carlo prices:")
            print(f"    Mean absolute error: {results['mean_mc_abs_error']:.6f}")
            print(f"    Mean relative error: {results['mean_mc_rel_error']:.6f}")
            print(f"    RMSE: {results['mc_rmse']:.6f}")
            print(f"    Speedup vs MC: {results['speedup_vs_mc']:.2f}x")
        
        print(f"  Performance:")
        print(f"    Avg prediction time per option: {results['prediction_time_per_option']:.6f} seconds")
    
    # Compile results
    evaluation_results = {
        'train_configs': train_configs,
        'train_prices': train_prices,
        'test_configs': test_configs,
        'test_prices': test_prices,
        'lstm_results': lstm_results
    }
    
    if mc_prices:
        evaluation_results.update({
            'mc_prices': mc_prices,
            'mc_times': mc_times
        })
    
    # Only plot if explicitly requested
    if plot:
        plot_evaluation_results(evaluation_results)
    
    return evaluation_results


# Create a modified plotting function with option to limit number of displayed options
def plot_evaluation_results(results, max_options=None):
    """
    Plot key evaluation metrics for LSTM models.
    
    Parameters:
    -----------
    results : dict
        Evaluation results from evaluate_lstm_model
    max_options : int, optional
        Maximum number of options to display in plots. If None, display all.
    """
    test_prices = results['test_prices']
    lstm_results = results['lstm_results']
    mc_prices = results.get('mc_prices', None)
    
    variant_names = list(lstm_results.keys())
    total_options = len(test_prices)
    
    # Determine how many options to plot
    if max_options is not None and max_options < total_options:
        # Select evenly spaced options if we need to limit
        indices = np.linspace(0, total_options-1, max_options, dtype=int)
        option_ids = [f"{i+1}" for i in indices]
        
        # Filter the data
        plot_test_prices = [test_prices[i] for i in indices]
        plot_mc_prices = [mc_prices[i] for i in indices] if mc_prices else None
        
        # Also filter the LSTM results
        plot_lstm_results = {}
        for name in variant_names:
            plot_lstm_results[name] = {
                'prices': [lstm_results[name]['prices'][i] for i in indices],
                'market_rel_errors': [lstm_results[name]['market_rel_errors'][i] for i in indices]
            }
            if mc_prices:
                plot_lstm_results[name]['mc_rel_errors'] = [lstm_results[name]['mc_rel_errors'][i] for i in indices]
        
        print(f"Showing {max_options} options out of {total_options} total options")
    else:
        # Use all options
        option_ids = [f"{i+1}" for i in range(total_options)]
        plot_test_prices = test_prices
        plot_mc_prices = mc_prices
        plot_lstm_results = {name: {
            'prices': lstm_results[name]['prices'],
            'market_rel_errors': lstm_results[name]['market_rel_errors'],
            'mc_rel_errors': lstm_results[name].get('mc_rel_errors', None)
        } for name in variant_names}
    
    # 1. Price Comparison
    plt.figure(figsize=(12, 6))
    
    plt.plot(option_ids, plot_test_prices, 'ko-', label='Market Price', linewidth=2)
    if plot_mc_prices:
        plt.plot(option_ids, plot_mc_prices, 'ro-', label='Monte Carlo Price', linewidth=2)
    
    for name in variant_names:
        plt.plot(option_ids, plot_lstm_results[name]['prices'], 'o--', label=name)
    
    plt.title('Price Comparison')
    plt.xlabel('Test Option ID')
    plt.ylabel('Option Price')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. Error Analysis
    plt.figure(figsize=(12, 6))
    
    # Market errors
    plt.subplot(1, 2, 1)
    for name in variant_names:
        plt.plot(option_ids, np.array(plot_lstm_results[name]['market_rel_errors']) * 100, 'o-', label=name)
    
    plt.title('Relative Error vs Market Prices (%)')
    plt.xlabel('Test Option ID')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)

    
    # MC errors if available
    if mc_prices:
        plt.subplot(1, 2, 2)
        for name in variant_names:
            plt.plot(option_ids, np.array(plot_lstm_results[name]['mc_rel_errors']) * 100, 'o-', label=name)
        
        plt.title('Relative Error vs Monte Carlo Prices (%)')
        plt.xlabel('Test Option ID')
        plt.ylabel('Relative Error (%)')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)

    
    plt.tight_layout()
    plt.show()
    
    # 3. Performance summary - using original full dataset for accuracy metrics
    plt.figure(figsize=(12, 6))
    
    # Error metrics
    x = np.arange(len(variant_names))
    width = 0.3
    
    market_rmse = [lstm_results[name]['market_rmse'] for name in variant_names]
    market_mae = [lstm_results[name]['mean_market_abs_error'] for name in variant_names]
    
    plt.bar(x - width/2, market_rmse, width, label='RMSE vs Market')
    plt.bar(x + width/2, market_mae, width, label='MAE vs Market')
    
    plt.xlabel('Model Variant')
    plt.ylabel('Error')
    plt.title('Error Metrics')
    plt.xticks(x, variant_names)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # 4. Training history
    plt.figure(figsize=(12, 6))
    
    for name in variant_names:
        history = lstm_results[name]['training_result']['history']
        plt.plot(history['loss'], label=f'{name} - Training')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], '--', label=f'{name} - Validation')
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to save evaluation results to disk
def save_evaluation_results(results, filename='lstm_evaluation_results.pkl'):
    """Save evaluation results to disk for later use."""
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")


# Function to load saved evaluation results
def load_evaluation_results(filename='lstm_evaluation_results.pkl'):
    """Load evaluation results from disk."""
    import pickle
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Results loaded from {filename}")
    return results


# Example usage with the new functions
def run_option_pricing_example(df, save_results=True, plot_now=False):
    """
    Run a complete example of LSTM option pricing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing option market data
    save_results : bool
        Whether to save results to disk
    plot_now : bool
        Whether to plot results immediately
    """
    # Create Monte Carlo pricer
    mc_pricer = MCPricer(n_sims=10000, n_steps=252)
    
    # Configure LSTM models
    lstm_configs = [
        {
            'name': 'Simple LSTM',
            'params': {
                'lstm_units': [32],
                'dropout_rate': 0.1,
                'bidirectional': False,
                'lookback_window': 10,
                'option_type': 'asian'
            }
        },
        {
            'name': 'Deep LSTM',
            'params': {
                'lstm_units': [64, 32, 16],
                'dropout_rate': 0.2,
                'bidirectional': False,
                'lookback_window': 20,
                'option_type': 'asian'
            }
        },
        {
            'name': 'Bidirectional LSTM',
            'params': {
                'lstm_units': [48, 24],
                'dropout_rate': 0.2,
                'bidirectional': True,
                'lookback_window': 15,
                'option_type': 'asian'
            }
        }
    ]

    simple_lstm_config = {
        'name': 'Simple LSTM',
        'params': {
            'lstm_units': [32],
            'dropout_rate': 0.1,
            'bidirectional': False,
            'lookback_window': 10,
            'option_type': 'asian'
        }
    }
    
    # Run evaluation without plotting
    results = evaluate_lstm_model(
        df,
        lstm_configs=lstm_configs,
        mc_pricer=mc_pricer,
        asian_option_class=AsianOption,
        option_type='asian',
        plot=plot_now
    )
    
    # Save results if requested
    if save_results:
        save_evaluation_results(results)
    
    return results


# Function to plot saved results
def plot_saved_results(filename='lstm_evaluation_results.pkl', max_options=20):
    """
    Plot results from a saved evaluation file with option to limit the number of displayed options.
    
    Parameters:
    -----------
    filename : str
        Path to the saved results file
    max_options : int or None
        Maximum number of options to display in plots. If None, display all.
    """
    results = load_evaluation_results(filename)
    plot_evaluation_results(results, max_options=max_options)
    return results