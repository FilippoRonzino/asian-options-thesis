# Asian Option Pricing: A Comparative Analysis of Analytical, Monte Carlo, and Deep Learning Methods
This repo contains the code used in my thesis for the BSc in Mathematical and Computing Sciences for Artificial Intelligence (BAI) at Bocconi University.

## Abstract
This thesis analyzes methods for pricing Asian options, path-dependent derivatives crucial for managing volatility risk. Due to their averaging feature, pricing, especially for common arithmetic Asian options, is challenging. We compare three approaches: analytical solutions (available only for geometric Asian options), Monte Carlo simulations with variance reduction techniques, and Deep Learning using Long Short-Term Memory (LSTM) networks trained on historical data. Results confirm analytical tractability only for geometric types and the efficiency of MC with control variates for arithmetic ones. Advanced LSTM models demonstrate competitive accuracy against benchmarks and market prices, suggesting their potential as fast, data-driven alternatives. The study concludes that while analytical methods are limited, enhanced Monte Carlo and modern Deep Learning techniques offer robust and promising tools for Asian option valuation.

## Structure
- `data/`: contains the fetchers and processors for the historical datasets used in the pricers.
- `plots/`: contains a notebook used to visualize some preliminary results.
- `src/`: contains the pricers and the main notebook used to run the simulations, it includes:
    - `constructors/`: contains the classes used to construct the `AsianOption` and `VanillaOption`, wrapper around a `BaseOption`;
    - `monte_carlo/`: contains both a `mc_pricer.py` and `test_mc_pricer.py` files, used to run the Monte Carlo simulations and test them;
    - `machine_learning/`: contains an instantiator for the `LSTMOptionPricer` and an `evaluator.py` file used to evaluate the models in the `main.py` comparing them with MC simulations;
- `test/`: contains the tests for the MC pricer on generated and historical data;

