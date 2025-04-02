import pandas as pd

def prepare_historical_data(df):
    """
    Prepare historical data for options by organizing by security description.
    
    :param df: DataFrame containing option market data
    :return: DataFrame with organized historical data
    """
    df_copy = df.copy()
    
    if 'Date' in df_copy.columns and not pd.api.types.is_datetime64_dtype(df_copy['Date']):
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    if 'EXPIRE_DT' in df_copy.columns and not pd.api.types.is_datetime64_dtype(df_copy['EXPIRE_DT']):
        df_copy['EXPIRE_DT'] = pd.to_datetime(df_copy['EXPIRE_DT'])
    
    df_sorted = df_copy.sort_values(['SECURITY_DES', 'Date'])
    
    if 'TIME_TO_MATURITY' not in df_sorted.columns and 'EXPIRE_DT' in df_sorted.columns and 'Date' in df_sorted.columns:
        df_sorted['TIME_TO_MATURITY'] = (df_sorted['EXPIRE_DT'] - df_sorted['Date']).dt.days / 365.0
    
    return df_sorted


def custom_prepare_option_data(df):
    """
    Custom function to prepare option configurations from market data DataFrame.
    
    :param df: DataFrame containing option market data
    :return: Tuple of option configurations and market prices
    """
    option_configs = []
    market_prices = []
    
    # get unique options by taking the most recent data point for each security description
    unique_options = df.sort_values('Date').groupby('SECURITY_DES').last().reset_index()
    
    filtered_df = unique_options[
        (unique_options['TIME_TO_MATURITY'] > 0) &
        (unique_options['OPT_PX'] > 0) &
        (unique_options['STRIKE_PX'] > 0) &
        (unique_options['UNDERLYING_PRICE'] > 0)
    ]
    
    for idx, row in filtered_df.iterrows():
        sigma = row['HIST_CALL_IMP_VOL']
        
        if pd.isna(sigma) or sigma <= 0:
            for vol_column in ['VOLATILITY_30D', 'VOLATILITY_20D', 'VOLATILITY_60D', 'VOLATILITY_90D', 'VOLATILITY_10D']:
                if vol_column in row and not pd.isna(row[vol_column]) and row[vol_column] > 0:
                    sigma = row[vol_column]
                    break
            
            if pd.isna(sigma) or sigma <= 0:
                continue
        
        if sigma > 1:
            sigma = sigma / 100
        
        config = {
            'name': row['SECURITY_DES'],  
            'S0': row['UNDERLYING_PRICE'],
            'K': row['STRIKE_PX'],
            'T': row['TIME_TO_MATURITY'],
            'r': row['RISK_FREE_RATE'],
            'sigma': sigma,
            'option_type': 'arithmetic'  # for Asian options
        }
        
        option_configs.append(config)
        market_prices.append(row['OPT_PX'])
    
    return option_configs, market_prices