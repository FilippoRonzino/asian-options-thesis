import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


class LSTMOptionPricer:
    """
    Neural network pricing model for options using LSTM architecture.
    """
    
    def __init__(self, lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001, 
                 bidirectional=False, lookback_window=20, option_type='asian'):
        """
        Initialize the LSTM option pricer.
        
        :param lstm_units: Number of units in each LSTM layer
        :param dropout_rate: Dropout rate for regularization
        :param learning_rate: Learning rate for Adam optimizer
        :param bidirectional: Whether to use bidirectional LSTM layers
        :param lookback_window: Number of time steps to look back for time series data
        :param option_type: Type of option to price ('regular', 'asian', etc.)
        :return: None
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
        """
        Build the LSTM model architecture.

        :param input_shape: Shape of the input data (timesteps, features)
        :return: Compiled LSTM model
        """
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
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model

    def _prepare_time_series(self, option_configs, prices=None, historical_data=None):
        """
        Convert option configurations to time series format for LSTM.
        
        :param option_configs: List of option configuration dictionaries
        :param prices: Option prices for training data
        :param historical_data: DataFrame containing historical time series data for options
        """
        features = []
        for config in option_configs:
            S0 = config['S0']
            K = config['K']
            T = config['T']
            r = config['r']
            sigma = config['sigma']
            
            # common derived features
            moneyness = S0 / K
            log_moneyness = np.log(moneyness)
            sigma_sqrt_t = sigma * np.sqrt(T)
            time_decay_factor = np.exp(-r * T)
            
            feature_vector = [S0, K, T, r, sigma, moneyness, log_moneyness, sigma_sqrt_t, time_decay_factor]
            features.append(feature_vector)
        
        features = np.array(features)
        if not self.is_trained:
            self.feature_scaler.fit(features)
        features_scaled = self.feature_scaler.transform(features)
        
        X = []
        
        # if we have historical data, use it
        if historical_data is not None:
            for i, config in enumerate(option_configs):
                security_des = config.get('name', None)
                if security_des is None:
                    continue
                    
                option_history = historical_data[historical_data['SECURITY_DES'] == security_des]
                option_history = option_history.sort_values('Date')
                
                if len(option_history) < self.lookback_window:
                    continue
                    
                recent_history = option_history.iloc[-self.lookback_window:]
                time_series = []
                
                for _, row in recent_history.iterrows():

                    current_S0 = row['UNDERLYING_PRICE']
                    current_K = config['K']  # Strike price doesn't change
                    current_T = row['TIME_TO_MATURITY']  
                    current_r = row['RISK_FREE_RATE']
                    
                    # Get implied volatility
                    current_sigma = row['HIST_CALL_IMP_VOL']
                    if pd.isna(current_sigma) or current_sigma <= 0:
                        # Fallback to alternative volatility metrics
                        for vol_column in ['VOLATILITY_30D', 'VOLATILITY_20D', 'VOLATILITY_60D', 'VOLATILITY_90D', 'VOLATILITY_10D']:
                            if vol_column in row and not pd.isna(row[vol_column]) and row[vol_column] > 0:
                                current_sigma = row[vol_column]
                                break
                    
                    if pd.isna(current_sigma) or current_sigma <= 0:
                        current_sigma = config['sigma']  # Fallback to provided sigma
                    
                    if current_sigma > 1:
                        current_sigma = current_sigma / 100
                    
                    current_moneyness = current_S0 / current_K
                    current_log_moneyness = np.log(current_moneyness)
                    current_sigma_sqrt_t = current_sigma * np.sqrt(current_T)
                    current_time_decay_factor = np.exp(-current_r * current_T)
                    
                    current_feature = [
                        current_S0, current_K, current_T, current_r, current_sigma,
                        current_moneyness, current_log_moneyness, current_sigma_sqrt_t, current_time_decay_factor
                    ]
                    
                    current_feature_scaled = self.feature_scaler.transform([current_feature])[0]
                    time_series.append(current_feature_scaled)
                
                if len(time_series) == self.lookback_window:
                    X.append(time_series)
        else:
            # fallback to the synthetic time series generation if no historical data is provided
            for i, feature in enumerate(features_scaled):
                time_series = []
                config = option_configs[i]
                
                for j in range(self.lookback_window):
                    time_frac = j / (self.lookback_window - 1)
                    adjusted_T = config['T'] * (1 - time_frac)
                    adjusted_feature = self._create_adjusted_feature(config, adjusted_T, time_frac)
                    adjusted_feature_scaled = self.feature_scaler.transform([adjusted_feature])[0]
                    time_series.append(adjusted_feature_scaled)
                
                X.append(time_series)
        
        X = np.array(X)
        
        if prices is not None:
            # if prices are provided, prepare target values
            prices = np.array(prices).reshape(-1, 1)
            
            if not self.is_trained:
                self.price_scaler.fit(prices)
            
            y = self.price_scaler.transform(prices)
            return X, y
        else:
            return X
    
    def _create_adjusted_feature(self, config, adjusted_T):
        """
        Create adjusted feature vector with new time to maturity.
        
        :param config: Option configuration dictionary
        :param adjusted_T: Adjusted time to maturity
        :return: Adjusted feature vector
        """
        S0 = config['S0']
        K = config['K']
        r = config['r']
        sigma = config['sigma']
        
        moneyness = S0 / K
        log_moneyness = np.log(moneyness)
        sigma_sqrt_t = sigma * np.sqrt(adjusted_T)
        adjusted_time_decay = np.exp(-r * adjusted_T)
        
        adjusted_feature = [S0, K, adjusted_T, r, sigma, moneyness, log_moneyness, sigma_sqrt_t, adjusted_time_decay]
        
        # Add Asian-specific features if needed
        # if self.option_type == 'asian':
        #     adjusted_sigma = sigma / np.sqrt(3)
        #     adjusted_sigma_sqrt_t = adjusted_sigma * np.sqrt(adjusted_T)
        #     adjusted_feature.extend([adjusted_sigma, adjusted_sigma_sqrt_t])
            
        return adjusted_feature
    
    def train(self, option_configs, prices, historical_data=None, validation_split=0.4, epochs=100, batch_size=32, verbose=1):
        """
        Train the LSTM model on option prices.
        
        :param option_configs: List of option configuration dictionaries
        :param prices: Option prices for training data
        :param historical_data: DataFrame containing historical time series data for options
        :param validation_split: Fraction of data to use for validation
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param verbose: Verbosity mode (0, 1, or 2)
        :return: Training history and time taken
        """
        X, y = self._prepare_time_series(option_configs, prices, historical_data)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        input_shape = (X.shape[1], X.shape[2])
        self.model = self._build_model(input_shape)
        
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
        """
        Predict option prices using the trained LSTM model.
        
        :param option_configs: List of option configuration dictionaries
        :param historical_data: DataFrame containing historical time series data for options
        :return: Predicted option prices and time taken for prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self._prepare_time_series(option_configs, historical_data=historical_data)
        
        start_time = time.time()
        y_pred_scaled = self.model.predict(X)
        prediction_time = time.time() - start_time
        
        y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten(), prediction_time

