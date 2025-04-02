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
                 bidirectional=False, lookback_window=20, option_type='asian'):
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

    def _prepare_time_series(self, option_configs, prices=None, historical_data=None):
        """Convert option configurations to time series format for LSTM.
        
        Parameters:
        -----------
        option_configs : list of dict
            List of option configuration dictionaries
        prices : list or ndarray, optional
            Option prices for training data
        historical_data : DataFrame, optional
            DataFrame containing historical time series data for options
        """
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
            features.append(feature_vector)
        
        # Scale features
        features = np.array(features)
        if not self.is_trained:
            self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        # Create time series data
        X = []
        
        # If we have historical data, use it
        if historical_data is not None:
            for i, config in enumerate(option_configs):
                security_des = config.get('name', None)
                if security_des is None:
                    # Skip if no security description
                    continue
                    
                # Filter historical data for this security
                option_history = historical_data[historical_data['SECURITY_DES'] == security_des]
                
                # Sort by date to ensure proper time ordering
                option_history = option_history.sort_values('Date')
                
                # Skip if not enough historical data
                if len(option_history) < self.lookback_window:
                    # Could either skip or use padding
                    continue
                    
                # Take the most recent lookback_window data points
                recent_history = option_history.iloc[-self.lookback_window:]
                
                # Create time series for this option
                time_series = []
                
                for _, row in recent_history.iterrows():
                    # Extract features from historical data
                    current_S0 = row['UNDERLYING_PRICE']
                    current_K = config['K']  # Strike price doesn't change
                    current_T = row['TIME_TO_MATURITY']  # Time to maturity decreases as we approach expiry
                    current_r = row['RISK_FREE_RATE']
                    
                    # Get implied volatility
                    current_sigma = row['HIST_CALL_IMP_VOL']
                    if pd.isna(current_sigma) or current_sigma <= 0:
                        # Fallback to alternative volatility metrics
                        for vol_column in ['VOLATILITY_30D', 'VOLATILITY_20D', 'VOLATILITY_60D', 'VOLATILITY_90D', 'VOLATILITY_10D']:
                            if vol_column in row and not pd.isna(row[vol_column]) and row[vol_column] > 0:
                                current_sigma = row[vol_column]
                                break
                    
                    # Skip if no valid volatility found
                    if pd.isna(current_sigma) or current_sigma <= 0:
                        current_sigma = config['sigma']  # Fallback to provided sigma
                    
                    # Convert volatility from percentage if needed
                    if current_sigma > 1:
                        current_sigma = current_sigma / 100
                    
                    # Calculate derived features
                    current_moneyness = current_S0 / current_K
                    current_log_moneyness = np.log(current_moneyness)
                    current_sigma_sqrt_t = current_sigma * np.sqrt(current_T)
                    current_time_decay_factor = np.exp(-current_r * current_T)
                    
                    # Create feature vector
                    current_feature = [
                        current_S0, current_K, current_T, current_r, current_sigma,
                        current_moneyness, current_log_moneyness, current_sigma_sqrt_t, current_time_decay_factor
                    ]
                    
                    # Scale this feature
                    current_feature_scaled = self.feature_scaler.transform([current_feature])[0]
                    time_series.append(current_feature_scaled)
                
                # Only add if we have a complete time series
                if len(time_series) == self.lookback_window:
                    X.append(time_series)
        else:
            # Fall back to the synthetic time series generation if no historical data is provided
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
    
    def train(self, option_configs, prices, historical_data=None, validation_split=0.4, epochs=100, batch_size=32, verbose=1):
        """Train the LSTM model on option prices."""
        # Prepare time series data
        X, y = self._prepare_time_series(option_configs, prices, historical_data)
        
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
    
    def predict_price(self, option_configs, historical_data=None):
        """Predict option prices using the trained LSTM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare time series data
        X = self._prepare_time_series(option_configs, historical_data=historical_data)
        
        # Predict
        start_time = time.time()
        y_pred_scaled = self.model.predict(X)
        prediction_time = time.time() - start_time
        
        # Inverse transform to get actual prices
        y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten(), prediction_time

def prepare_historical_data(df):
    """
    Prepare historical data for options by organizing by security description.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing option market data with time series
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all historical data
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Ensure the Date column is in datetime format
    if 'Date' in df_copy.columns and not pd.api.types.is_datetime64_dtype(df_copy['Date']):
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Ensure EXPIRE_DT is in datetime format if it exists
    if 'EXPIRE_DT' in df_copy.columns and not pd.api.types.is_datetime64_dtype(df_copy['EXPIRE_DT']):
        df_copy['EXPIRE_DT'] = pd.to_datetime(df_copy['EXPIRE_DT'])
    
    # Sort by security description and date
    df_sorted = df_copy.sort_values(['SECURITY_DES', 'Date'])
    
    # Calculate time to maturity if not already present and if we have the necessary columns
    if 'TIME_TO_MATURITY' not in df_sorted.columns and 'EXPIRE_DT' in df_sorted.columns and 'Date' in df_sorted.columns:
        df_sorted['TIME_TO_MATURITY'] = (df_sorted['EXPIRE_DT'] - df_sorted['Date']).dt.days / 365.0
    
    return df_sorted


def custom_prepare_option_data(df):
    """Custom function to prepare option configurations from market data DataFrame."""
    option_configs = []
    market_prices = []
    
    # Get unique options by taking the most recent data point for each security description
    unique_options = df.sort_values('Date').groupby('SECURITY_DES').last().reset_index()
    
    # Filter for call options and valid data
    filtered_df = unique_options[
        (unique_options['TIME_TO_MATURITY'] > 0) &
        (unique_options['OPT_PX'] > 0) &
        (unique_options['STRIKE_PX'] > 0) &
        (unique_options['UNDERLYING_PRICE'] > 0)
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
            'name': row['SECURITY_DES'],  # Make sure to include the security description
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
                        option_type='asian', plot=False, cv=True, save_data=True):
    """
    Evaluate LSTM models against market data and Monte Carlo pricing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing option market data with time series data
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
    cv : bool
        Whether to use control variates in Monte Carlo pricing
    save_data : bool
        Whether to save processed data to CSV files
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
    
    # Prepare historical data
    historical_data = prepare_historical_data(df)
    
    # Prepare option data using our custom function
    option_configs, market_prices = custom_prepare_option_data(df)
    
    if len(option_configs) == 0:
        raise ValueError(f"No valid {option_type} option data found after filtering")
    
    print(f"Prepared {len(option_configs)} valid option configurations from market data")
    
    # Split data - ensure we're splitting by unique security descriptions
    unique_securities = list({config['name'] for config in option_configs})
    train_securities, test_securities = train_test_split(
        unique_securities, test_size=0.4, random_state=42
    )
    
    # Create training and test sets based on security descriptions
    train_configs = [config for config in option_configs if config['name'] in train_securities]
    train_prices = [market_prices[i] for i, config in enumerate(option_configs) if config['name'] in train_securities]
    
    test_configs = [config for config in option_configs if config['name'] in test_securities]
    test_prices = [market_prices[i] for i, config in enumerate(option_configs) if config['name'] in test_securities]
    
    print(f"Training set: {len(train_configs)} options, Test set: {len(test_configs)} options")
    
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
            price = mc_pricer.price(option, use_control_variate=cv)
            mc_time = time.time() - start_time
            
            mc_prices.append(price)
            mc_times.append(mc_time)
    
    # Save processed data if requested
    if save_data:
        print("\nSaving processed data to CSV files...")
        save_processed_data(
            historical_data=historical_data,
            option_configs=option_configs,
            market_prices=market_prices,
            train_configs=train_configs,
            train_prices=train_prices,
            test_configs=test_configs,
            test_prices=test_prices,
            mc_prices=mc_prices if mc_prices else None
        )
    
    # Train and evaluate each LSTM model
    lstm_results = {}
    
    for config in lstm_configs:
        name = config['name']
        params = config['params']
        
        print(f"\nTraining {name}...")
        
        # Create and train the model
        lstm_pricer = LSTMOptionPricer(**params)
        plot_time_series_diagnostics(lstm_pricer, option_configs, historical_data)

        
        # Pass the historical data to the training function
        training_result = lstm_pricer.train(
            train_configs, 
            train_prices, 
            historical_data=historical_data,
            epochs=100, 
            verbose=1
        )
        
        # Save a sample of LSTM input data if requested
        if save_data:
            print(f"Saving sample LSTM input data for {name}...")
            save_lstm_input_dataset(
                lstm_pricer=lstm_pricer,
                option_configs=test_configs,
                historical_data=historical_data,
                base_filename=f"lstm_input_{name.lower().replace(' ', '_')}"
            )
        
        # Predict prices using historical data
        lstm_prices, prediction_time = lstm_pricer.predict_price(
            test_configs,
            historical_data=historical_data
        )
        
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
    plt.figure(figsize=(20, 8))
    
    # Market errors
    plt.subplot(1, 2, 1)
    for name in variant_names:
        plt.plot(option_ids, np.array(plot_lstm_results[name]['market_rel_errors']) * 100, 'o-', label=name)
    
    plt.title('Relative Error vs Market Prices (%)')
    plt.xlabel('Test Option ID')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=90)

    
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
        plt.xticks(rotation=90)

    
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
def run_option_pricing_example(df, save_results=True, plot_now=True, save_data=True):
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
    save_data : bool
        Whether to save processed data to CSV files
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
                'lookback_window': 20,
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
                'lookback_window': 20,
                'option_type': 'asian'
            }
        }
    ]
    
    # Run evaluation with saving data if requested
    results = evaluate_lstm_model(
        df,
        lstm_configs=lstm_configs,
        mc_pricer=mc_pricer,
        asian_option_class=AsianOption,
        option_type='asian',
        plot=plot_now,
        save_data=save_data
    )
    
    # Save MC option details if requested
    if save_data and mc_pricer is not None and 'test_configs' in results:
        save_mc_option_details(
            results['test_configs'],
            mc_pricer,
            AsianOption
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


def save_processed_data(historical_data, option_configs, market_prices, train_configs, train_prices, 
                        test_configs, test_prices, mc_prices=None, base_filename="option_data"):
    """
    Save all processed data structures to CSV files for inspection.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        Processed historical data from prepare_historical_data()
    option_configs : list of dict
        Option configuration dictionaries
    market_prices : list
        Market prices for all options
    train_configs : list of dict
        Training set option configurations
    train_prices : list
        Training set market prices
    test_configs : list of dict
        Test set option configurations
    test_prices : list
        Test set market prices
    mc_prices : list, optional
        Monte Carlo prices for test set if available
    base_filename : str
        Base name for output files
    """
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs("processed_data", exist_ok=True)
    
    # 1. Save historical data
    historical_data.to_csv(f"processed_data/{base_filename}_historical_data.csv", index=False)
    print(f"Saved historical data to processed_data/{base_filename}_historical_data.csv")
    
    # 2. Save option configurations and prices
    # Convert list of dictionaries to DataFrame
    config_df = pd.DataFrame(option_configs)
    config_df['market_price'] = market_prices
    config_df.to_csv(f"processed_data/{base_filename}_all_options.csv", index=False)
    print(f"Saved all option configurations to processed_data/{base_filename}_all_options.csv")
    
    # 3. Save training data
    train_df = pd.DataFrame(train_configs)
    train_df['market_price'] = train_prices
    train_df.to_csv(f"processed_data/{base_filename}_train_options.csv", index=False)
    print(f"Saved training options to processed_data/{base_filename}_train_options.csv")
    
    # 4. Save test data
    test_df = pd.DataFrame(test_configs)
    test_df['market_price'] = test_prices
    
    if mc_prices is not None:
        test_df['mc_price'] = mc_prices
    
    test_df.to_csv(f"processed_data/{base_filename}_test_options.csv", index=False)
    print(f"Saved test options to processed_data/{base_filename}_test_options.csv")
    
    return {
        "historical_data_path": f"processed_data/{base_filename}_historical_data.csv",
        "all_options_path": f"processed_data/{base_filename}_all_options.csv",
        "train_options_path": f"processed_data/{base_filename}_train_options.csv",
        "test_options_path": f"processed_data/{base_filename}_test_options.csv"
    }


def save_lstm_input_dataset(lstm_pricer, option_configs, historical_data, base_filename="lstm_input_dataset"):
    """
    Save the entire processed LSTM input dataset to CSV for inspection.

    Parameters:
    -----------
    lstm_pricer : LSTMOptionPricer
        Trained LSTM pricer instance
    option_configs : list of dict
        Option configurations to prepare
    historical_data : pandas.DataFrame
        Historical data DataFrame
    base_filename : str
        Base name for output file
    """
    import pandas as pd
    import numpy as np
    import os

    # Create output directory if it doesn't exist
    os.makedirs("processed_data", exist_ok=True)

    # Prepare time series data for all options
    X = lstm_pricer._prepare_time_series(option_configs, historical_data=historical_data)

    # Create a DataFrame to store the entire dataset
    all_data = []

    for i, time_series in enumerate(X):
        # Create DataFrame from the time series
        feature_names = [
            'S0', 'K', 'T', 'r', 'sigma', 'moneyness',
            'log_moneyness', 'sigma_sqrt_t', 'time_decay_factor'
        ]

        # If the data is scaled, we need to inverse transform it
        if lstm_pricer.is_trained:
            time_series_unscaled = lstm_pricer.feature_scaler.inverse_transform(time_series)
        else:
            time_series_unscaled = time_series

        # Create DataFrame
        ts_df = pd.DataFrame(time_series_unscaled, columns=feature_names)

        # Add time step index
        ts_df['time_step'] = range(len(ts_df))

        # Add option details
        for key, value in option_configs[i].items():
            if key not in ts_df.columns and not isinstance(value, dict) and not isinstance(value, list):
                ts_df[f'config_{key}'] = value

        # Append to the list
        all_data.append(ts_df)

    # Concatenate all data into a single DataFrame
    all_data_df = pd.concat(all_data, keys=range(len(all_data)))

    # Save to CSV
    filename = f"processed_data/{base_filename}.csv"
    all_data_df.to_csv(filename, index=True)
    print(f"Saved entire LSTM input dataset to {filename}")

    return filename


def save_mc_option_details(test_configs, mc_pricer, asian_option_class, base_filename="mc_option_details"):
    """
    Save Monte Carlo option objects to CSV with all relevant parameters.

    Parameters:
    -----------
    test_configs : list of dict
        Test set option configurations
    mc_pricer : MCPricer
        Monte Carlo pricer instance
    asian_option_class : class
        Asian option class
    base_filename : str
        Base name for output file
    """
    import pandas as pd
    import os

    # Create output directory if it doesn't exist
    os.makedirs("processed_data", exist_ok=True)

    mc_options_data = []

    print("Creating detailed Monte Carlo option data...")
    for i, config in enumerate(test_configs):
        # Create option object
        option = asian_option_class(
            config['S0'], config['K'], config['T'],
            config['r'], config['sigma'],
            n_steps=252, option_type=config['option_type']
        )

        # Get all option parameters and Monte Carlo config
        option_data = {
            'option_id': i+1,
            'security_name': config.get('name', f'Option_{i+1}'),
            'S0': option.S0,
            'K': option.K,
            'T': option.T,
            'r': option.r,
            'sigma': option.sigma,
            'option_type': option.option_type,
            'n_steps': option.n_steps,
            'mc_n_sims': mc_pricer.n_sims,
            'mc_n_steps': mc_pricer.n_steps,
            'random_number_generator': str(mc_pricer.__class__.__name__),
        }

        # Add pricing results
        price = mc_pricer.price(option, use_control_variate=True)
        price_no_cv = mc_pricer.price(option, use_control_variate=False)

        option_data.update({
            'mc_price_with_cv': price,
            'mc_price_no_cv': price_no_cv,
            'mc_price_difference': price - price_no_cv,
        })

        mc_options_data.append(option_data)

    # Save to CSV
    mc_df = pd.DataFrame(mc_options_data)
    filename = f"processed_data/{base_filename}.csv"
    mc_df.to_csv(filename, index=False)
    print(f"Saved Monte Carlo option details to {filename}")

    return filename


def inspect_saved_data(csv_path, show_head=True, show_info=True, show_stats=True):
    """
    Load and inspect a saved CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    show_head : bool
        Whether to show the first few rows
    show_info : bool
        Whether to show DataFrame info
    show_stats : bool
        Whether to show descriptive statistics
        
    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame
    """
    import pandas as pd
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    print(f"Loaded data from {csv_path}")
    print(f"Shape: {df.shape} - {df.shape[0]} rows, {df.shape[1]} columns")
    
    if show_head:
        print("\nFirst few rows:")
        print(df.head())
    
    if show_info:
        print("\nColumn information:")
        print(df.info())
    
    if show_stats:
        print("\nDescriptive statistics:")
        print(df.describe())
    
    return df


def compare_time_series_data(time_series_csvs, key_columns=None):
    """
    Compare the most populated time series from multiple CSVs.
    
    Parameters:
    -----------
    time_series_csvs : list
        List of paths to time series CSV files
    key_columns : list
        List of column names to focus on (if None, use all numeric columns except 'time_step' and 'config_name')
        
    Returns:
    --------
    dict
        Dictionary of filtered DataFrames
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    dfs = {}
    filtered_dfs = {}
    
    for i, path in enumerate(time_series_csvs):
        df = pd.read_csv(path)
        name = f"Series_{i+1}"
        dfs[name] = df
        
        print(f"{name} (from {path}): Shape {df.shape}")
        
        # Identify the most populated config_name
        top_config = df['config_name'].value_counts().idxmax()
        df_filtered = df[df['config_name'] == top_config]
        filtered_dfs[name] = df_filtered
        
        print(f"  Most populated config_name: {top_config} ({len(df_filtered)} rows)")
        
        if key_columns is None:
            key_columns = [col for col in df.columns if col not in ['time_step', 'config_name']]
        
        # Show min/max for key columns
        for col in key_columns:
            if col in df.columns:
                print(f"  {col}: min={df_filtered[col].min():.4f}, max={df_filtered[col].max():.4f}, mean={df_filtered[col].mean():.4f}")
        print()
    
    # Plot time series for key columns
    n_cols = len(key_columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3*n_cols))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(key_columns):
        ax = axes[i]
        for name, df in filtered_dfs.items():
            if col in df.columns and 'time_step' in df.columns:
                ax.plot(df['time_step'], df[col], label=name)
        
        ax.set_title(f'{col} over Time for Most Populated Config')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return filtered_dfs


def compare_lstm_mc_predictions(results_csv, plot=True):
    """
    Compare LSTM and MC price predictions against market prices.
    
    Parameters:
    -----------
    results_csv : str
        Path to the test options CSV with LSTM and MC prices
    plot : bool
        Whether to plot comparisons
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparisons
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the data
    df = pd.read_csv(results_csv)
    
    # Check if we have the required columns
    required_cols = ['market_price']
    lstm_col = next((col for col in df.columns if 'lstm' in col.lower()), None)
    mc_col = next((col for col in df.columns if 'mc_price' in col.lower()), None)
    
    if not all(col in df.columns for col in required_cols) or not lstm_col and not mc_col:
        print(f"CSV doesn't contain required price columns. Found columns: {df.columns.tolist()}")
        return df
    
    # Calculate errors
    if lstm_col:
        df['lstm_abs_error'] = np.abs(df[lstm_col] - df['market_price'])
        df['lstm_rel_error'] = df['lstm_abs_error'] / df['market_price'] * 100
    
    if mc_col:
        df['mc_abs_error'] = np.abs(df[mc_col] - df['market_price'])
        df['mc_rel_error'] = df['mc_abs_error'] / df['market_price'] * 100
        
        if lstm_col:
            df['lstm_mc_diff'] = np.abs(df[lstm_col] - df[mc_col])
            df['lstm_mc_rel_diff'] = df['lstm_mc_diff'] / df[mc_col] * 100
    
    # Show summary statistics
    print("Price comparison summary:")
    stats_cols = ['market_price']
    if lstm_col:
        stats_cols.extend([lstm_col, 'lstm_abs_error', 'lstm_rel_error'])
    if mc_col:
        stats_cols.extend([mc_col, 'mc_abs_error', 'mc_rel_error'])
    if lstm_col and mc_col:
        stats_cols.extend(['lstm_mc_diff', 'lstm_mc_rel_diff'])
    
    print(df[stats_cols].describe())
    
    # Plot if requested
    if plot:
        n_options = len(df)
        x = np.arange(n_options)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x, df['market_price'], 'ko-', label='Market Price', linewidth=2)
        
        if lstm_col:
            plt.plot(x, df[lstm_col], 'bo--', label='LSTM Price')
        
        if mc_col:
            plt.plot(x, df[mc_col], 'ro--', label='MC Price')
        
        plt.title('Price Comparison')
        plt.xlabel('Option Index')
        plt.ylabel('Option Price')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Error comparison
        if lstm_col or mc_col:
            plt.figure(figsize=(12, 6))
            
            if lstm_col:
                plt.plot(x, df['lstm_rel_error'], 'bo-', label='LSTM Relative Error (%)')
            
            if mc_col:
                plt.plot(x, df['mc_rel_error'], 'ro-', label='MC Relative Error (%)')
            
            plt.title('Relative Error Comparison')
            plt.xlabel('Option Index')
            plt.ylabel('Relative Error (%)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    return df


def plot_time_series_diagnostics(lstm_pricer, option_configs, historical_data, num_options=2):
    """
    Plot time series data for diagnostic purposes.

    Parameters:
    -----------
    lstm_pricer : LSTMOptionPricer
        Trained LSTM pricer instance
    option_configs : list of dict
        Option configurations to prepare
    historical_data : pandas.DataFrame
        Historical data DataFrame
    num_options : int
        Number of options to plot for diagnostics
    """
    import matplotlib.pyplot as plt

    # Prepare time series data
    X = lstm_pricer._prepare_time_series(option_configs, historical_data=historical_data)

    # Plot time series for the first few options
    for i in range(min(num_options, len(X))):
        time_series = X[i]
        time_steps = range(len(time_series))

        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, time_series[:, 0], label='S0')
        plt.plot(time_steps, time_series[:, 2], label='T')
        plt.plot(time_steps, time_series[:, 4], label='sigma')
        plt.title(f'Time Series for Option {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
