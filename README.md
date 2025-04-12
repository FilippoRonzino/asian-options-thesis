# Asian Options Pricing: From Analytical Solutions to Monte Carlo Simulations and Machine Learning
This repo contains the code used in my thesis for the BSc in Mathematical and Computing Sciences for Artificial Intelligence (BAI) at Bocconi University.

## Abstract
TODO

## Structure
- `data/`: contains the fetchers and processors for the historical datasets used in the pricers.
- `plots/`: contains a notebook used to visualize some preliminary results.
- `src/`: contains the pricers and the main notebook used to run the simulations, it includes:
    - `constructors/`: contains the classes used to construct the `AsianOption` and `VanillaOption`, wrapper around a `BaseOption`;
    - `monte_carlo/`: contains both a `mc_pricer.py` and `test_mc_pricer.py` files, used to run the Monte Carlo simulations and test them;
    - `machine_learning/`: contains an instantiator for the `LSTMOptionPricer` and an `evaluator.py` file used to evaluate the models in the `main.py` comparing them with MC simulations;
- `test/`: contains the tests for the MC pricer on generated and historical data;

