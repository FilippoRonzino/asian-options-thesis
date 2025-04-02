import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from machine_learning.data_preparation import (
    custom_prepare_option_data,
    prepare_historical_data,
)
from machine_learning.lstm_option_pricer import LSTMOptionPricer
from constructors.asianoption import AsianOption
from monte_carlo.mcpricer import MCPricer


def evaluate_lstm_model(df, lstm_configs=None, mc_pricer=None, asian_option_class=None, 
                        option_type='asian', plot=False, cv=True, save_data=True):
    """
    Evaluate LSTM models against market data and Monte Carlo pricing.
    
    :param df: DataFrame containing option market data
    :param lstm_configs: List of LSTM model configurations
    :param mc_pricer: Monte Carlo pricer instance
    :param asian_option_class: Asian option class for pricing
    :param option_type: Type of option to price ('regular', 'asian', etc.)
    :param plot: Whether to plot evaluation results
    :param cv: Whether to use control variate for Monte Carlo pricing
    :param save_data: Whether to save processed data to CSV files
    :return: Evaluation results including LSTM model performance
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
    
    historical_data = prepare_historical_data(df)
    option_configs, market_prices = custom_prepare_option_data(df)
    
    if len(option_configs) == 0:
        raise ValueError(f"No valid {option_type} option data found after filtering")
    
    print(f"Prepared {len(option_configs)} valid option configurations from market data")
    
    # splitting by unique security descriptions
    unique_securities = list({config['name'] for config in option_configs})
    train_securities, test_securities = train_test_split(
        unique_securities, test_size=0.4, random_state=42
    )
    
    # training and test sets based on security descriptions
    train_configs = [config for config in option_configs if config['name'] in train_securities]
    train_prices = [market_prices[i] for i, config in enumerate(option_configs) if config['name'] in train_securities]
    
    test_configs = [config for config in option_configs if config['name'] in test_securities]
    test_prices = [market_prices[i] for i, config in enumerate(option_configs) if config['name'] in test_securities]
    
    print(f"Training set: {len(train_configs)} options, Test set: {len(test_configs)} options")
    
    mc_prices = []
    mc_times = []
    
    if mc_pricer is not None and asian_option_class is not None:
        print("Generating Monte Carlo prices for comparison...")
        for i, config in enumerate(test_configs):
            print(f"Processing option {i+1}/{len(test_configs)}")
            
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
    
    lstm_results = {}
    
    for config in lstm_configs:
        name = config['name']
        params = config['params']
        
        print(f"\nTraining {name}...")
        
        lstm_pricer = LSTMOptionPricer(**params)
        plot_time_series_diagnostics(lstm_pricer, option_configs, historical_data)

        training_result = lstm_pricer.train(
            train_configs, 
            train_prices, 
            historical_data=historical_data,
            epochs=100, 
            verbose=1
        )
        
        if save_data:
            print(f"Saving sample LSTM input data for {name}...")
            save_lstm_input_dataset(
                lstm_pricer=lstm_pricer,
                option_configs=test_configs,
                historical_data=historical_data,
                base_filename=f"lstm_input_{name.lower().replace(' ', '_')}"
            )
        
        lstm_prices, prediction_time = lstm_pricer.predict_price(
            test_configs,
            historical_data=historical_data
        )
        
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
        
        print(f"{name} test results:")
        print("  Against market prices:")
        print(f"    Mean absolute error: {results['mean_market_abs_error']:.6f}")
        print(f"    Mean relative error: {results['mean_market_rel_error']:.6f}")
        print(f"    RMSE: {results['market_rmse']:.6f}")
        
        if mc_prices:
            print("  Against Monte Carlo prices:")
            print(f"    Mean absolute error: {results['mean_mc_abs_error']:.6f}")
            print(f"    Mean relative error: {results['mean_mc_rel_error']:.6f}")
            print(f"    RMSE: {results['mc_rmse']:.6f}")
            print(f"    Speedup vs MC: {results['speedup_vs_mc']:.2f}x")
        
        print("  Performance:")
        print(f"    Avg prediction time per option: {results['prediction_time_per_option']:.6f} seconds")
    
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
    
    if plot:
        plot_evaluation_results(evaluation_results)
    
    return evaluation_results


def plot_evaluation_results(results, max_options=None):
    """
    Plot key evaluation metrics for LSTM models.
    
    :param results: Evaluation results dictionary
    :param max_options: Maximum number of options to display in plots. If None, display all.
    :return: None
    """
    test_prices = results['test_prices']
    lstm_results = results['lstm_results']
    mc_prices = results.get('mc_prices', None)
    
    variant_names = list(lstm_results.keys())
    total_options = len(test_prices)
    
    if max_options is not None and max_options < total_options:
        # select evenly spaced options if we need to limit
        indices = np.linspace(0, total_options-1, max_options, dtype=int)
        option_ids = [f"{i+1}" for i in indices]
        
        plot_test_prices = [test_prices[i] for i in indices]
        plot_mc_prices = [mc_prices[i] for i in indices] if mc_prices else None
        
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

def save_evaluation_results(results, filename='lstm_evaluation_results.pkl'):
    """Save evaluation results to disk for later use."""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")


def load_evaluation_results(filename='lstm_evaluation_results.pkl'):
    """Load evaluation results from disk."""    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Results loaded from {filename}")
    return results


def run_option_pricing_example(df, save_results=True, plot_now=True, save_data=True):
    """
    Run a complete example of LSTM option pricing.
    
    :param df: DataFrame containing option market data
    :param save_results: Whether to save the evaluation results
    :param plot_now: Whether to plot results immediately
    :param save_data: Whether to save processed data to CSV files
    :return: Evaluation results
    """
    mc_pricer = MCPricer(n_sims=10000, n_steps=252)
    
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
    
    results = evaluate_lstm_model(
        df,
        lstm_configs=lstm_configs,
        mc_pricer=mc_pricer,
        asian_option_class=AsianOption,
        option_type='asian',
        plot=plot_now,
        save_data=save_data
    )
    
    if save_data and mc_pricer is not None and 'test_configs' in results:
        save_mc_option_details(
            results['test_configs'],
            mc_pricer,
            AsianOption
        )
    
    if save_results:
        save_evaluation_results(results)
    
    return results


def plot_saved_results(filename='lstm_evaluation_results.pkl', max_options=20):
    """
    Plot results from a saved evaluation file with option to limit the number of displayed options.
    
    :param filename: Path to the saved evaluation results file
    :param max_options: Maximum number of options to display in plots
    :return: Evaluation results
    """
    results = load_evaluation_results(filename)
    plot_evaluation_results(results, max_options=max_options)
    return results


def save_processed_data(historical_data, option_configs, market_prices, train_configs, train_prices, 
                        test_configs, test_prices, mc_prices=None, base_filename="option_data"):
    """
    Save all processed data structures to CSV files for inspection.
    
    :param historical_data: DataFrame containing historical data
    :param option_configs: List of option configurations
    :param market_prices: List of market prices for options
    :param train_configs: List of training option configurations
    :param train_prices: List of training option prices
    :param test_configs: List of test option configurations
    :param test_prices: List of test option prices
    :param mc_prices: List of Monte Carlo prices (optional)
    :param base_filename: Base name for output files
    :return: Dictionary with paths to saved files
    """
    os.makedirs("processed_data", exist_ok=True)
    
    # 1. Save historical data
    historical_data.to_csv(f"processed_data/{base_filename}_historical_data.csv", index=False)
    print(f"Saved historical data to processed_data/{base_filename}_historical_data.csv")
    
    # 2. Save option configurations and prices
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

    :param lstm_pricer: LSTMOptionPricer instance
    :param option_configs: List of option configurations
    :param historical_data: DataFrame containing historical data
    :param base_filename: Base name for output file
    :return: Path to the saved CSV file
    """
    os.makedirs("processed_data", exist_ok=True)

    X = lstm_pricer._prepare_time_series(option_configs, historical_data=historical_data)
    all_data = []

    for i, time_series in enumerate(X):
        feature_names = [
            'S0', 'K', 'T', 'r', 'sigma', 'moneyness',
            'log_moneyness', 'sigma_sqrt_t', 'time_decay_factor'
        ]

        if lstm_pricer.is_trained:
            time_series_unscaled = lstm_pricer.feature_scaler.inverse_transform(time_series)
        else:
            time_series_unscaled = time_series

        ts_df = pd.DataFrame(time_series_unscaled, columns=feature_names)

        ts_df['time_step'] = range(len(ts_df))

        for key, value in option_configs[i].items():
            if key not in ts_df.columns and not isinstance(value, dict) and not isinstance(value, list):
                ts_df[f'config_{key}'] = value

        all_data.append(ts_df)

    all_data_df = pd.concat(all_data, keys=range(len(all_data)))

    filename = f"processed_data/{base_filename}.csv"
    all_data_df.to_csv(filename, index=True)
    print(f"Saved entire LSTM input dataset to {filename}")

    return filename


def save_mc_option_details(test_configs, mc_pricer, asian_option_class, base_filename="mc_option_details"):
    """
    Save Monte Carlo option objects to CSV with all relevant parameters.

    :param test_configs: List of option configurations for testing
    :param mc_pricer: Monte Carlo pricer instance
    :param asian_option_class: Instance of AsianOption class
    :param base_filename: Base name for output file
    :return: Path to the saved CSV file
    """
    os.makedirs("processed_data", exist_ok=True)

    mc_options_data = []

    print("Creating detailed Monte Carlo option data...")
    for i, config in enumerate(test_configs):
        option = asian_option_class(
            config['S0'], config['K'], config['T'],
            config['r'], config['sigma'],
            n_steps=252, option_type=config['option_type']
        )

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

        price = mc_pricer.price(option, use_control_variate=True)
        price_no_cv = mc_pricer.price(option, use_control_variate=False)

        option_data.update({
            'mc_price_with_cv': price,
            'mc_price_no_cv': price_no_cv,
            'mc_price_difference': price - price_no_cv,
        })

        mc_options_data.append(option_data)

    mc_df = pd.DataFrame(mc_options_data)
    filename = f"processed_data/{base_filename}.csv"
    mc_df.to_csv(filename, index=False)
    print(f"Saved Monte Carlo option details to {filename}")

    return filename


def inspect_saved_data(csv_path, show_head=True, show_info=True, show_stats=True):
    """
    Load and inspect a saved CSV file.
    
    :param csv_path: Path to the CSV file
    :param show_head: Whether to show the first few rows
    :param show_info: Whether to show column information
    :param show_stats: Whether to show descriptive statistics
    :return: DataFrame containing the loaded data
    """    
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
    
    :param time_series_csvs: List of paths to CSV files containing time series data
    :param key_columns: List of key columns to compare (optional)
    :return: Dictionary of filtered DataFrames
    """
    dfs = {}
    filtered_dfs = {}
    
    for i, path in enumerate(time_series_csvs):
        df = pd.read_csv(path)
        name = f"Series_{i+1}"
        dfs[name] = df
        
        print(f"{name} (from {path}): Shape {df.shape}")
        
        # most populated config_name
        top_config = df['config_name'].value_counts().idxmax()
        df_filtered = df[df['config_name'] == top_config]
        filtered_dfs[name] = df_filtered
        
        print(f"  Most populated config_name: {top_config} ({len(df_filtered)} rows)")
        
        if key_columns is None:
            key_columns = [col for col in df.columns if col not in ['time_step', 'config_name']]
        
        for col in key_columns:
            if col in df.columns:
                print(f"  {col}: min={df_filtered[col].min():.4f}, max={df_filtered[col].max():.4f}, mean={df_filtered[col].mean():.4f}")
        print()
    
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
    
    :param results_csv: Path to the CSV file containing LSTM and MC predictions
    :param plot: Whether to plot the results
    :return: DataFrame containing the loaded data and error metrics
    """
    df = pd.read_csv(results_csv)
    
    required_cols = ['market_price']
    lstm_col = next((col for col in df.columns if 'lstm' in col.lower()), None)
    mc_col = next((col for col in df.columns if 'mc_price' in col.lower()), None)
    
    if not all(col in df.columns for col in required_cols) or not lstm_col and not mc_col:
        print(f"CSV doesn't contain required price columns. Found columns: {df.columns.tolist()}")
        return df
    
    if lstm_col:
        df['lstm_abs_error'] = np.abs(df[lstm_col] - df['market_price'])
        df['lstm_rel_error'] = df['lstm_abs_error'] / df['market_price'] * 100
    
    if mc_col:
        df['mc_abs_error'] = np.abs(df[mc_col] - df['market_price'])
        df['mc_rel_error'] = df['mc_abs_error'] / df['market_price'] * 100
        
        if lstm_col:
            df['lstm_mc_diff'] = np.abs(df[lstm_col] - df[mc_col])
            df['lstm_mc_rel_diff'] = df['lstm_mc_diff'] / df[mc_col] * 100
    
    print("Price comparison summary:")
    stats_cols = ['market_price']
    if lstm_col:
        stats_cols.extend([lstm_col, 'lstm_abs_error', 'lstm_rel_error'])
    if mc_col:
        stats_cols.extend([mc_col, 'mc_abs_error', 'mc_rel_error'])
    if lstm_col and mc_col:
        stats_cols.extend(['lstm_mc_diff', 'lstm_mc_rel_diff'])
    
    print(df[stats_cols].describe())
    
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

    :param lstm_pricer: LSTMOptionPricer instance
    :param option_configs: List of option configurations
    :param historical_data: DataFrame containing historical data
    :param num_options: Number of options to plot
    :return: None
    """
    X = lstm_pricer._prepare_time_series(option_configs, historical_data=historical_data)

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
