from evaluation import evaluate_lstm_model, save_evaluation_results, save_mc_option_details
from constructors.asianoption import AsianOption
from monte_carlo.mc_pricer import MCPricer


def run_option_pricing_example(df, save_results=True, plot_now=True, save_data=True):
    """
    Run a complete example of LSTM option pricing.
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
