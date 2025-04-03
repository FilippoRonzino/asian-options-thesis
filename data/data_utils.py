def get_ticker_by_value_count(df, strike, value_count):
    """
    Get the ticker with the most occurrences in the dataframe for a given strike price. 
    Filter by the ones that have at least value_count occurrences.

    :param df: DataFrame containing the data.
    :param strike: The strike price to filter by.
    :param value_count: The minimum number of occurrences for a ticker to be included.
    :return: List of tickers that have at least value_count occurrences for the given strike price.
    """
    filtered_df = df[df['STRIKE_PX'] == strike]
    ticker_counts = filtered_df['SECURITY_DES'].value_counts()
    print(f"Selecting tickers with at least {value_count} occurrences.")

    return list(ticker_counts[ticker_counts >= value_count].index)

def filter_by_ticker_and_strike(df, ticker, strike):
    """
    Filter the dataframe by ticker and strike price.

    :param df: DataFrame containing the data.
    :param ticker: The ticker to filter by.
    :param strike: The strike price to filter by.
    :return: Filtered DataFrame.
    """
    security_des = f"{ticker} {strike} Comdty"
    return df[df['SECURITY_DES'] == security_des]

def filter_by_security_des(df, security_des):
    """
    Filter the dataframe by SECURITY_DES.

    :param df: DataFrame containing the data.
    :param security_des: The SECURITY_DES to filter by.
    :return: Filtered DataFrame.
    """
    return df[df['SECURITY_DES'] == security_des]

def filter_by_strike(df, strike):
    """
    Filter the dataframe by STRIKE_PX.

    :param df: DataFrame containing the data.
    :param strike: The strike price to filter by.
    :return: Filtered DataFrame.
    """
    return df[df['STRIKE_PX'] == strike]